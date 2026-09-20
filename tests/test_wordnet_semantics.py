import io
import json
from urllib.error import HTTPError, URLError

import pytest

from spoken_to_signed.text_to_gloss import wordnet


def test_hypernyms_instances_and_cycles(monkeypatch):
    calls = []
    graph = {
        "day": {"hypernym": {"data": [{"id": "period"}]}},
        "period": {"hypernym": {"data": [{"id": "omw-en-15113229-n"}]}},
        "holiday": {"instance_hypernym": {"data": [{"id": "day"}]}},
        "loop": {"hypernym": {"data": [{"id": "loop"}]}},
        "other": {},
    }

    def fetch(url, timeout):
        assert timeout == 5
        name = url.rsplit("/", 1)[-1]
        calls.append(name)
        return io.BytesIO(json.dumps({"data": {"relationships": graph[name]}}).encode())

    monkeypatch.setattr(wordnet, "urlopen", fetch)
    wn = wordnet.WordNet("http://wordnet")
    assert wn.is_time("holiday")
    assert wn.is_time("holiday")
    assert calls == ["holiday", "day", "period"]
    assert not wn.is_time("loop")
    assert not wn.is_time("other")
    assert wn.is_time("omw-en-00507716-r")


def test_failed_requests_are_not_cached(monkeypatch):
    calls = []

    def fetch(url, timeout):
        calls.append(url)
        raise URLError("offline")

    monkeypatch.setattr(wordnet, "urlopen", fetch)
    wn = wordnet.WordNet("http://wordnet")
    for _ in range(2):
        with pytest.raises(wordnet.WordNetUnavailableError, match="lookup failed"):
            wn.is_time("example")
    assert len(calls) == 2


def test_clock_times_have_a_separate_measurement_ancestry(monkeypatch):
    # Verified via the WordNet API: noon -> hour -> time-of-day (a reading),
    # not the time-period / point-in-time roots.
    parents = {"omw-en-15165490-n": ("omw-en-15228378-n",),
               "omw-en-15228378-n": ("omw-en-15129927-n",), "omw-en-15129927-n": ()}
    wn = wordnet.WordNet("http://wordnet")
    monkeypatch.setattr(wn, "parents", lambda sense: parents[sense])
    assert wn.is_time("omw-en-15165490-n")


def test_unknown_sense_is_not_an_outage(monkeypatch):
    def fetch(url, timeout):
        raise HTTPError(url, 404, "not found", {}, None)

    monkeypatch.setattr(wordnet, "urlopen", fetch)
    assert not wordnet.WordNet("http://wordnet").is_time("wikidata-en-L1-S1")


def test_ids_cannot_replace_the_configured_host(monkeypatch):
    urls = []

    def fetch(url, timeout):
        urls.append(url)
        return io.BytesIO(b'{"data":{"relationships":{}}}')

    monkeypatch.setattr(wordnet, "urlopen", fetch)
    assert not wordnet.WordNet("http://wordnet").is_time("https://other/?secret")
    assert urls == ["http://wordnet/lexicons/omw-en:1.4/synsets/https%3A%2F%2Fother%2F%3Fsecret"]


def test_shared_deadline_bounds_multiple_traversals(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(wordnet, "monotonic", lambda: clock[0])
    wn = wordnet.WordNet("http://wordnet")

    def parents(synset):
        clock[0] += 6
        return ("omw-en-15113229-n",)

    monkeypatch.setattr(wn, "parents", parents)
    assert wn.is_time("first", deadline=10)
    with pytest.raises(wordnet.WordNetUnavailableError, match="time budget"):
        wn.is_time("second", deadline=10)


def test_missing_pinned_resource_fails(monkeypatch):
    def fetch(url, timeout):
        raise HTTPError(url, 404, "not found", {}, None)

    monkeypatch.setattr(wordnet, "urlopen", fetch)
    with pytest.raises(wordnet.WordNetUnavailableError):
        wordnet.WordNet("http://wordnet").parents("omw-en-15113229-n")
