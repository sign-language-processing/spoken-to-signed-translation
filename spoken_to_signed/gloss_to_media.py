"""Resolve annotated glosses through dictionary-api, then fingerspell misses.

Dictionary visibility belongs to the API. Only complete-span senses are queried;
contained senses in ``source`` never accidentally translate part of an entity.
"""

import os
from collections import defaultdict
from functools import lru_cache
from io import BytesIO
from threading import Lock
from typing import Optional

import httpx
from pydantic import BaseModel, ConfigDict, StrictInt


class Link(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    id: StrictInt
    confidence: float
    wikidata_id: Optional[str]
    wordnet_synset_id: Optional[str]
    bucket_url: Optional[str]
    signwriting: Optional[str]


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


def identifiers(token, field):
    if field == "wikidata_id":
        return [
            f"Q{item['id']}" if str(item["id"]).isdigit() else str(item["id"]) for item in token.get("entities", [])
        ]
    return [str(item["id"]) for item in token.get("synsets", [])]


class Dictionary:
    def __init__(self, client: httpx.Client, url: str):
        self.client, self.url = client, url.rstrip("/")

    def links(self, ids, signed_language):
        if not any(ids.values()):
            return []
        try:
            response = self.client.post(
                f"{self.url}/internal/links",
                json={
                    "wikidata_ids": ids["wikidata_id"],
                    "wordnet_synset_ids": ids["wordnet_synset_id"],
                    "signed_language": signed_language,
                },
            )
            response.raise_for_status()
            body = response.json()
            if body.get("success") is not True or not isinstance(body.get("data"), list):
                raise ValueError("Invalid dictionary envelope")
            rows = [Link.model_validate(item) for item in body["data"]]
            if any(not any(getattr(row, field) in values for field, values in ids.items()) for row in rows):
                raise ValueError("Dictionary returned an unrequested concept")
            return [row for row in rows if row.confidence > 0]
        except (httpx.HTTPError, ValueError, AttributeError) as error:
            raise LookupUnavailableError("Dictionary lookup failed or returned invalid assets") from error

    def candidates(self, tokens, signed_language):
        fields = ("wikidata_id", "wordnet_synset_id")
        ids = {field: sorted({value for token in tokens for value in identifiers(token, field)}) for field in fields}
        if any(len(values) > 1024 for values in ids.values()):
            raise ValueError("At most 1024 distinct entity IDs and 1024 sense IDs per batch")
        by_concept = defaultdict(list)
        for link in self.links(ids, signed_language):
            for field in fields:
                if getattr(link, field) in ids[field]:
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


class Realizer:
    def __init__(self, dictionary: Dictionary):
        self.dictionary = dictionary

    def resolve(self, tokens, target, spoken_language, signed_language, fingerspelling=True):
        candidates = self.dictionary.candidates(tokens, signed_language)
        results = []
        for token, hits in zip(tokens, candidates):
            if target == "video":
                path = next((hit.bucket_url for hit in hits if hit.bucket_url), None)
                results.append({"bucket_url": path} if path else {"text": token.get("word") or token["gloss"]})
                continue
            value = next((hit.signwriting for hit in hits if hit.signwriting), None)
            # TODO: the SignWriting fingerspelling library has no batch API yet.
            if not value and fingerspelling:
                value = spell(token, "signwriting", spoken_language, signed_language)
            if not value:
                raise MissingSignError(f"No SignWriting for {token.get('word') or token['gloss']!r}")
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
        # Keep mutable pose caches request-local: concatenation transforms pose views in place.
        pose = (
            FingerspellingPoseLookup(reduce=False)
            .lookup(
                letters,
                token["gloss"],
                spoken_language,
                signed_language,
            )
            .pose
        )
        buffer = BytesIO()
        pose.write(buffer)
        return buffer.getvalue()
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
        return Realizer(Dictionary(client, url)).resolve(
            tokens,
            target,
            spoken_language,
            signed_language,
            fingerspelling,
        )
