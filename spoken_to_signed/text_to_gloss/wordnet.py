"""Small semantic adapter to the existing WordNet API; no word re-disambiguation."""

from __future__ import annotations

import json
from functools import lru_cache
from time import monotonic
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import urlopen

# OMW English 1.4 semantic roots; descendants match too.
TIME_ROOTS = {
    "omw-en-15113229-n",  # Time period: an amount of time.
    "omw-en-15180528-n",  # Point in time: an instant.
    "omw-en-15154774-n",  # Time unit: a unit for measuring time periods.
    "omw-en-15129927-n",  # Clock time: a time-of-day reading, e.g. 10 o'clock.
}
# WordNet adverbs have no hypernym hierarchy. These are exact selected senses,
# not word triggers.
TIME_ADVERBS = {
    "omw-en-00507716-r",  # Yesterday: on the preceding day.
    "omw-en-00507819-r",  # Yesterday: in the recent past (figurative sense).
    "omw-en-00479275-r",  # Tomorrow: on the following day.
    "omw-en-00207366-r",  # Today: on this day (literal sense).
    "omw-en-00079499-r",  # Tonight: during the present day's night.
}
VOLITION_ROOTS = {
    "omw-en-01825237-v",  # Want, desire: feel or have a strong desire for something.
    "omw-en-00708538-v",  # Intend, mean: have in mind as a purpose.
    "omw-en-02530167-v",  # Try, attempt, seek: make an effort to do something.
}
SEMANTIC_ROOTS = {"time": TIME_ROOTS | TIME_ADVERBS, "volition": VOLITION_ROOTS}


class WordNetUnavailableError(RuntimeError):
    pass


class WordNet:
    def __init__(self, url: str):
        if urlparse(url).scheme not in {"http", "https"}:
            raise ValueError("WordNet URL must use http or https")
        self.url = url.rstrip("/")
        # Bound the cache per adapter. Exceptions are deliberately not cached.
        self.parents = lru_cache(maxsize=8192)(self._parents)

    def _parents(self, synset: str, *, timeout: float = 5) -> tuple[str, ...]:
        url = f"{self.url}/lexicons/omw-en:1.4/synsets/{quote(synset, safe='')}"
        try:
            with urlopen(url, timeout=timeout) as response:
                payload = response.read(1024 * 1024 + 1)
                if len(payload) > 1024 * 1024:
                    raise WordNetUnavailableError("WordNet response exceeded size limit")
                relations = json.loads(payload)["data"]["relationships"]
            return tuple(
                item["id"]
                for relation in ("hypernym", "instance_hypernym")
                for item in relations.get(relation, {}).get("data", [])
            )
        except HTTPError as error:
            if error.code == 404 and synset not in TIME_ROOTS | TIME_ADVERBS | VOLITION_ROOTS:
                return ()  # A valid but unsupported sense has no known ancestry.
            raise WordNetUnavailableError("WordNet semantic lookup failed") from error
        except (URLError, TimeoutError, ValueError, KeyError, TypeError) as error:
            raise WordNetUnavailableError("WordNet semantic lookup failed") from error

    def matches(self, synset: str, category: str = "time", *, deadline: float | None = None) -> bool:
        roots = SEMANTIC_ROOTS[category]
        deadline = deadline if deadline is not None else monotonic() + 10
        pending, seen = [synset], set()
        while pending:
            if monotonic() >= deadline:
                raise WordNetUnavailableError("WordNet semantic lookup exceeded time budget")
            current = pending.pop()
            if current in roots:
                return True
            if current in seen:
                continue
            seen.add(current)
            if len(seen) > 128:
                raise WordNetUnavailableError("WordNet ancestry exceeded traversal limit")
            pending.extend(self.parents(current))
        return False
