"""Resolve annotated glosses through dictionary-api, then fingerspell misses.

Dictionary visibility belongs to the API. Only complete-span senses are queried;
contained senses in ``source`` never accidentally translate part of an entity.
"""

import base64
import os
from collections import defaultdict
from functools import lru_cache
from io import BytesIO
from threading import Lock
from typing import Optional

import httpx
from pydantic import BaseModel, ConfigDict, Field, StrictInt


class Assets(BaseModel):
    video_md5: Optional[str] = Field(pattern=r"^[a-f0-9]{32}$")
    signwriting: Optional[str]


class Link(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    id: StrictInt
    confidence: float
    wikidata_id: Optional[str]
    wordnet_synset_id: Optional[str]
    assets: Assets


class LookupUnavailableError(Exception):
    """An upstream failure, not a missing dictionary entry."""


class MissingSignError(Exception):
    """Neither a dictionary asset nor complete fingerspelling is available."""


_identity_lock = Lock()


@lru_cache(maxsize=1)
def _identity_credentials(audience):
    from google.oauth2.id_token import fetch_id_token_credentials

    return fetch_id_token_credentials(audience)


def identity_token(audience):
    from google.auth.transport.requests import Request

    # Google owns expiry/skew; serialize refreshes across request threads.
    with _identity_lock:
        credentials = _identity_credentials(audience)
        if not credentials.valid:
            credentials.refresh(Request())
        return credentials.token


@lru_cache(maxsize=1)
def storage_client():
    from google.cloud import storage

    return storage.Client()


def identifiers(token, field):
    if field == "wikidata_id":
        return [f"Q{item['id']}" if str(item["id"]).isdigit() else str(item["id"])
                for item in token.get("entities", [])]
    return [str(item["id"]) for item in token.get("synsets", [])]


class Dictionary:
    def __init__(self, client: httpx.Client, url: str):
        self.client, self.url = client, url.rstrip("/")

    def links(self, field, ids, signed_language):
        try:
            for start in range(0, len(ids), 100):
                response = self.client.get(f"{self.url}/internal/links", params={
                    field: ",".join(ids[start:start + 100]), "signed_language": signed_language,
                })
                response.raise_for_status()
                body = response.json()
                if body.get("success") is not True or not isinstance(body.get("data"), list):
                    raise ValueError("Invalid dictionary envelope")
                for row in body["data"]:
                    link = Link.model_validate(row)
                    if getattr(link, field) not in ids[start:start + 100]:
                        raise ValueError("Dictionary returned an unrequested concept")
                    if link.confidence > 0:
                        yield link
        except (httpx.HTTPError, ValueError, AttributeError) as error:
            raise LookupUnavailableError("Dictionary lookup failed or returned invalid assets") from error

    def candidates(self, tokens, signed_language):
        by_concept = defaultdict(list)
        for field in ("wikidata_id", "wordnet_synset_id"):
            ids = sorted({identifier for token in tokens for identifier in identifiers(token, field)})
            for link in self.links(field, ids, signed_language):
                by_concept[field, getattr(link, field)].append(link)
        result = []
        for token in tokens:
            hits, seen = [], set()
            for field in ("wikidata_id", "wordnet_synset_id"):
                rows = [row for identifier in identifiers(token, field) for row in by_concept[field, identifier]]
                for row in sorted(rows, key=lambda row: (-row.confidence, row.id)):
                    if row.id not in seen:
                        hits.append(row)
                        seen.add(row.id)
            result.append(hits)
        return result


class PoseStore:
    """Check metadata only. The gateway owns materializing MD5-addressed poses."""

    def __init__(self, bucket):
        self.bucket_name, self.bucket = bucket, None

    def exists(self, md5):
        from google.api_core.exceptions import GoogleAPICallError
        from google.auth.exceptions import GoogleAuthError

        try:
            if self.bucket is None:
                self.bucket = storage_client().bucket(self.bucket_name)
            return self.bucket.blob(f"videos/{md5}/holistic.pose").exists(timeout=10)
        except (GoogleAPICallError, GoogleAuthError) as error:
            raise LookupUnavailableError("Pose storage unavailable") from error


class Realizer:
    def __init__(self, dictionary: Dictionary, pose_store=None):
        self.dictionary, self.pose_store = dictionary, pose_store

    def pose_candidate(self, hits, exists):
        for hit in hits:
            if not (md5 := hit.assets.video_md5):
                continue
            if self.pose_store is None:
                raise LookupUnavailableError("Configure TRANSFORMED_BUCKET for dictionary poses")
            if md5 not in exists:
                exists[md5] = self.pose_store.exists(md5)
            if exists[md5]:
                return {"md5": md5}
        return None

    def resolve(self, tokens, target, spoken_language, signed_language, fingerspelling=True):
        candidates = self.dictionary.candidates(tokens, signed_language)
        results, exists = [], {}
        generated_bytes = 0
        # TODO: the fingerspelling libraries have scalar APIs; keep assembly serial for now.
        for token, hits in zip(tokens, candidates):
            value = (next((hit.assets.signwriting for hit in hits if hit.assets.signwriting), None)
                     if target == "signwriting" else self.pose_candidate(hits, exists))
            if not value and fingerspelling:
                value = spell(token, target, spoken_language, signed_language)
            if not value:
                raise MissingSignError(f"No {target} for {token.get('word') or token['gloss']!r}")
            if target == "pose" and "base64" in value:
                generated_bytes += len(value["base64"])
                if generated_bytes > 32 * 1024 * 1024:
                    raise ValueError("Generated pose payload exceeds 32 MiB; send a smaller batch")
            results.append(value)
        return results


def spell(token, target, spoken_language, signed_language):
    word = token.get("word") or token["gloss"]
    try:
        if target == "signwriting":
            from signwriting.fingerspelling.fingerspelling import spell_text

            return spell_text(word, language=signed_language, seed=0)
        from spoken_to_signed.gloss_to_pose.lookup.fingerspelling_lookup import FingerspellingPoseLookup

        # Spaces separate words, not letters. Unsupported symbols must fail, not disappear.
        letters = "".join(word.split())
        if not letters:
            return None
        pose = FingerspellingPoseLookup(reduce=False).lookup(
            letters, token["gloss"], spoken_language, signed_language,
        ).pose
        buffer = BytesIO()
        pose.write(buffer)
        return {"base64": base64.b64encode(buffer.getvalue()).decode("ascii")}
    except FileNotFoundError:
        return None


def realize(tokens, target, spoken_language, signed_language, fingerspelling=True):
    if not tokens:
        return []
    url = os.environ.get("DICTIONARY_API_URL")
    if not url:
        raise LookupUnavailableError("Configure DICTIONARY_API_URL to enable gloss realization")
    headers = {}
    if os.environ.get("SKIP_AUTH") != "true":
        from google.auth.exceptions import GoogleAuthError

        try:
            headers["Authorization"] = f"Bearer {identity_token(url)}"
        except GoogleAuthError as error:
            raise LookupUnavailableError("Dictionary identity unavailable") from error
    with httpx.Client(headers=headers, timeout=30) as client:
        bucket = os.environ.get("TRANSFORMED_BUCKET")
        store = PoseStore(bucket) if target == "pose" and bucket else None
        return Realizer(Dictionary(client, url), store).resolve(
            tokens, target, spoken_language, signed_language, fingerspelling,
        )
