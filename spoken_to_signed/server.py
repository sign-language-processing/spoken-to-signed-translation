"""HTTP adapters for pretokenized glossing and optional dictionary-based poses."""

import os
from hashlib import sha256
from io import BytesIO
from typing import Literal, Optional, Union

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, StrictInt

from spoken_to_signed.text_to_gloss import rules, simple
from spoken_to_signed.text_to_gloss.senses import prepare_tokens
from spoken_to_signed.text_to_gloss.types import GlossItem

MODEL_VERSION = os.environ.get("MODEL_VERSION", "")
if MODEL_VERSION:
    # A local model override must not attest the production GPT configuration.
    model_config = (os.environ.get("OPENAI_MODEL", "gpt-5.6-luna"), os.environ.get("OPENAI_BASE_URL", ""))
    MODEL_VERSION += "-" + sha256(repr(model_config).encode()).hexdigest()[:12]
app = FastAPI(title="Spoken-to-signed glossing")


class Token(BaseModel):
    # Carry caller annotations through glossing and on to downstream lookup.
    model_config = ConfigDict(extra="allow")

    word: Optional[str] = None
    gloss: str
    pos: Optional[str] = None
    morphology: list[dict[str, str]] = Field(default_factory=list)


class GlosserOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spoken_language: str
    signed_language: str
    glosser: Literal["rules", "simple", "gpt"] = "rules"


class GlossRequest(GlosserOptions):
    tokens: list[Token]


class SourceToken(BaseModel):
    model_config = ConfigDict(extra="allow")
    word: str
    lemma: str
    pos: str
    morph: dict[str, str] = Field(default_factory=dict)


class SenseSpan(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: Union[str, int]
    start_token: StrictInt
    end_token: StrictInt


class Senses(BaseModel):
    model_config = ConfigDict(extra="allow")
    tokens: list[SourceToken]
    synsets: list[SenseSpan]
    entities: list[SenseSpan]


class SensesRequest(GlosserOptions):
    senses: Senses


class GlossResponse(BaseModel):
    sentences: list[list[Token]]
    # Original input-item positions, grouped exactly like sentences.
    indexes: list[list[int]]


class PoseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokens: list[Token] = Field(min_length=1)
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


@app.post("/tokens-to-gloss", response_model=GlossResponse, response_model_exclude_unset=True)
def tokens_to_gloss(request: GlossRequest, response: Response):
    tokens = [GlossItem(token.word, token.gloss) for token in request.tokens]
    indexes = {id(token): index for index, token in enumerate(tokens)}
    if request.glosser == "gpt":
        from spoken_to_signed.text_to_gloss import gpt as glosser
    else:
        glosser = {"rules": rules, "simple": simple}[request.glosser]
    try:
        sentences = glosser.tokens_to_gloss(
            tokens,
            language=request.spoken_language,
            signed_language=request.signed_language,
            metadata=[token.model_dump(include={"pos", "morphology"}) for token in request.tokens],
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    response.headers["X-Model-Tag"] = MODEL_VERSION
    order = [[indexes[id(token)] for token in sentence] for sentence in sentences]
    return GlossResponse(sentences=[[request.tokens[index] for index in sentence] for sentence in order], indexes=order)


@app.post("/senses-to-gloss", response_model=GlossResponse, response_model_exclude_unset=True)
def senses_to_gloss(request: SensesRequest, response: Response):
    try:
        tokens = prepare_tokens(request.senses.model_dump(exclude_unset=True))
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return tokens_to_gloss(GlossRequest(tokens=tokens, **request.model_dump(exclude={"senses"})), response)


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
