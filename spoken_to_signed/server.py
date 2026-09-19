"""HTTP adapters for pretokenized glossing and optional dictionary-based poses."""

import os
from contextlib import asynccontextmanager
from functools import partial
from hashlib import sha256
from io import BytesIO
from time import monotonic
from typing import Literal, Optional, Union

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Receive, Scope, Send

from spoken_to_signed.text_to_gloss.senses import senses_to_gloss as gloss_senses
from spoken_to_signed.text_to_gloss.types import GlossItem
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
        # Reject a WordNet deployment without the pinned OMW resource at startup.
        await run_in_threadpool(semantics.parents, "omw-en-15113229-n")
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


class PoseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokens: list[Token] = Field(min_length=1, max_length=1024)
    spoken_language: str
    signed_language: str
    source: Optional[str] = None
    fingerspelling: bool = True
    anonymize: bool = False


def pose_lookup(fingerspelling: bool, source: Optional[str]):
    from spoken_to_signed.gloss_to_pose.lookup import CSVPoseLookup
    from spoken_to_signed.gloss_to_pose.lookup.fingerspelling_lookup import FingerspellingPoseLookup
    from spoken_to_signed.gloss_to_pose.lookup.sql_lookup import SQLPoseLookup

    backup = FingerspellingPoseLookup() if fingerspelling else None
    if database_url := os.environ.get("DATABASE_URL"):
        return SQLPoseLookup({"dsn": database_url}, backup=backup)
    if lexicon := os.environ.get("LEXICON_PATH"):
        if source is not None:
            raise HTTPException(status_code=422, detail="source filtering requires the PostgreSQL backend")
        return CSVPoseLookup(lexicon, backup=backup)
    raise HTTPException(status_code=503, detail="Configure DATABASE_URL or LEXICON_PATH to enable pose lookup")


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


@app.post("/gloss-to-pose", response_class=Response)
def gloss_to_pose(request: PoseRequest):
    from spoken_to_signed.gloss_to_pose import gloss_to_pose as construct_pose

    # A lookup per request keeps fallback settings and coverage request-local.
    lookup = pose_lookup(request.fingerspelling, request.source)
    tokens = [GlossItem(token.word or token.gloss, token.gloss) for token in request.tokens if token.pos != "PUNCT"]
    if not tokens:
        raise HTTPException(status_code=422, detail="No non-punctuation glosses")
    result = construct_pose(
        tokens,
        lookup,
        request.spoken_language,
        request.signed_language,
        source=request.source,
        anonymize=request.anonymize,
    )
    buffer = BytesIO()
    result.pose.write(buffer)
    return Response(buffer.getvalue(), media_type="application/pose", headers={"X-Model-Tag": MODEL_VERSION})
