"""HTTP adapters for pretokenized glossing and optional dictionary-based poses."""

import os
from contextlib import asynccontextmanager
from functools import partial
from hashlib import sha256
from time import monotonic
from typing import Literal, Optional, Union

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Receive, Scope, Send

from spoken_to_signed.gloss_to_media import LookupUnavailableError, MissingSignError, realize
from spoken_to_signed.text_to_gloss.senses import senses_to_gloss as gloss_senses
from spoken_to_signed.text_to_gloss.wordnet import WordNet, WordNetUnavailableError

MODEL_VERSION = os.environ.get("MODEL_VERSION", "")
semantics = WordNet(os.environ["WORDNET_URL"]) if os.environ.get("WORDNET_URL") else None
# Cache identity must distinguish semantic ordering from the offline fallback.
if MODEL_VERSION:
    MODEL_VERSION += "-" + sha256(os.environ.get("WORDNET_URL", "offline").encode()).hexdigest()[:12]


class BodyLimit:
    """Bound streamed bodies as well as requests with Content-Length."""

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        size = 0

        async def bounded_receive():
            nonlocal size
            message = await receive()
            size += len(message.get("body", b""))
            if size > 2 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Request body exceeds 2 MiB")
            return message

        await self.app(scope, bounded_receive if scope["type"] == "http" else receive, send)


@asynccontextmanager
async def lifespan(app):
    if semantics:
        # Validate the pinned resource, allowing a scale-to-zero WordNet service to wake up.
        await run_in_threadpool(semantics.parents, "omw-en-15113229-n", timeout=60)
    yield


app = FastAPI(title="Spoken-to-signed glossing", lifespan=lifespan)
app.add_middleware(BodyLimit)


class Token(BaseModel):
    # Carry caller annotations through glossing and on to downstream lookup.
    model_config = ConfigDict(extra="allow")

    word: Optional[str] = None
    gloss: str
    pos: Optional[str] = None
    morphology: list[dict[str, str]] = Field(default_factory=list)


class SourceToken(BaseModel):
    model_config = ConfigDict(extra="allow")
    word: str
    lemma: str
    pos: str
    dep: str
    head: StrictInt
    morph: dict[str, str] = Field(default_factory=dict)


class Span(BaseModel):
    model_config = ConfigDict(extra="allow")
    start_token: StrictInt
    end_token: StrictInt


class SenseSpan(Span):
    id: Union[str, int]


class SynsetSpan(Span):
    id: str


class Senses(BaseModel):
    model_config = ConfigDict(extra="allow")
    tokens: list[SourceToken] = Field(max_length=1024)
    synsets: list[SynsetSpan] = Field(max_length=1024)
    entities: list[SenseSpan] = Field(max_length=1024)
    sentences: list[Span] = Field(max_length=1024)


class SensesRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spoken_language: Literal["en"]
    signed_language: Literal["ase"]
    senses: Senses


class GlossResponse(BaseModel):
    sentences: list[list[Token]]
    # Positions in prepare_tokens(senses), not in the raw WSD token list.
    indexes: list[list[int]]
    changes: list[dict]
    notes: list[dict]


class Concept(BaseModel):
    id: Union[str, StrictInt]

    @field_validator("id")
    @classmethod
    def valid_identifier(cls, value):
        if not str(value) or len(str(value)) > 256 or any(c.isspace() or c == "," for c in str(value)):
            raise ValueError("Invalid concept identifier")
        return value


class MediaToken(Token):
    word: Optional[str] = Field(default=None, min_length=1, max_length=128)
    gloss: str = Field(min_length=1, max_length=128)
    synsets: list[Concept] = Field(default_factory=list, max_length=128)
    entities: list[Concept] = Field(default_factory=list, max_length=128)

    @field_validator("pos")
    @classmethod
    def not_punctuation(cls, value):
        if value == "PUNCT":
            raise ValueError("Punctuation is not a realizable gloss; use the current senses-to-gloss endpoint")
        return value


class MediaRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tokens: list[MediaToken] = Field(max_length=128)
    spoken_language: str = Field(pattern=r"^[a-z]{2,3}$")
    signed_language: str = Field(pattern=r"^[a-z]{3}$")
    fingerspelling: bool = True


@app.get("/health")
def health(response: Response):
    response.headers["X-Model-Tag"] = MODEL_VERSION
    return {"status": "healthy", "version": MODEL_VERSION}


@app.post("/senses-to-gloss", response_model=GlossResponse, response_model_exclude_unset=True)
def senses_to_gloss(request: SensesRequest, response: Response):
    try:
        is_time = partial(semantics.is_time, deadline=monotonic() + 10) if semantics else None
        result = gloss_senses(request.senses.model_dump(exclude_unset=True), semantics=is_time)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except WordNetUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    response.headers["X-Model-Tag"] = MODEL_VERSION
    return result


def realize_glosses(request: MediaRequest, response: Response, target: str):
    try:
        result = realize(
            [token.model_dump() for token in request.tokens],
            target,
            request.spoken_language,
            request.signed_language,
            request.fingerspelling,
        )
    except LookupUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except MissingSignError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    # Dictionary edits are not captured by the service's code version.
    response.headers.update({"X-Model-Tag": MODEL_VERSION, "Cache-Control": "no-store"})
    return result


@app.post("/gloss-to-pose")
def gloss_to_pose(request: MediaRequest, response: Response):
    return {"poses": realize_glosses(request, response, "pose")}


@app.post("/gloss-to-signwriting")
def gloss_to_signwriting(request: MediaRequest, response: Response):
    return {"signwriting": realize_glosses(request, response, "signwriting")}
