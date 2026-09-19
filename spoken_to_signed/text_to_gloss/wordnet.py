"""Small semantic adapter to the existing WordNet API; no word re-disambiguation."""

import json
from functools import lru_cache
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import urlopen

# OMW English 1.4: time period, point in time, time unit. IDs, not display labels.
TIME_ROOTS = {"omw-en-15113229-n", "omw-en-15180528-n", "omw-en-15154774-n"}
# WordNet adverbs have no hypernym hierarchy. These are exact selected senses,
# not word triggers: yesterday (literal/recent), tomorrow, today (literal), tonight.
TIME_ADVERBS = {"omw-en-00507716-r", "omw-en-00507819-r", "omw-en-00479275-r",
                "omw-en-00207366-r", "omw-en-00079499-r"}


class WordNetUnavailable(RuntimeError):
    pass


class WordNet:
    def __init__(self, url: str):
        if urlparse(url).scheme not in {"http", "https"}:
            raise ValueError("WordNet URL must use http or https")
        self.url = url.rstrip("/")
        # Bound the cache per adapter. Exceptions are deliberately not cached.
        self.parents = lru_cache(maxsize=8192)(self._parents)

    def _parents(self, synset: str) -> tuple[str, ...]:
        url = f"{self.url}/lexicons/omw-en:1.4/synsets/{quote(synset, safe='')}"
        try:
            with urlopen(url, timeout=5) as response:
                relations = json.load(response)["data"]["relationships"]
            return tuple(item["id"] for relation in ("hypernym", "instance_hypernym")
                         for item in relations.get(relation, {}).get("data", []))
        except HTTPError as error:
            if error.code == 404:
                return ()  # A valid but unsupported sense has no known ancestry.
            raise WordNetUnavailable("WordNet semantic lookup failed") from error
        except (URLError, TimeoutError, ValueError, KeyError, TypeError) as error:
            raise WordNetUnavailable("WordNet semantic lookup failed") from error

    def is_time(self, synset: str) -> bool:
        pending, seen = [synset], set()
        while pending:
            current = pending.pop()
            if current in TIME_ROOTS or current in TIME_ADVERBS:
                return True
            if current in seen:
                continue
            seen.add(current)
            if len(seen) > 128:
                raise WordNetUnavailable("WordNet ancestry exceeded traversal limit")
            pending.extend(self.parents(current))
        return False
