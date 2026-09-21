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
    assert wn.matches("holiday")
    assert wn.matches("holiday")
    assert calls == ["holiday", "day", "period"]
    assert not wn.matches("loop")
    assert not wn.matches("other")
    assert wn.matches("omw-en-00507716-r")


def test_failed_requests_are_not_cached(monkeypatch):
    calls = []

    def fetch(url, timeout):
        calls.append(url)
        raise URLError("offline")

    monkeypatch.setattr(wordnet, "urlopen", fetch)
    wn = wordnet.WordNet("http://wordnet")
    for _ in range(2):
        with pytest.raises(wordnet.WordNetUnavailableError, match="lookup failed"):
            wn.matches("example")
    assert len(calls) == 2


def test_clock_times_have_a_separate_measurement_ancestry(monkeypatch):
    # Verified via the WordNet API: noon -> hour -> time-of-day (a reading),
    # not the time-period / point-in-time roots.
    parents = {"omw-en-15165490-n": ("omw-en-15228378-n",),
               "omw-en-15228378-n": ("omw-en-15129927-n",), "omw-en-15129927-n": ()}
    wn = wordnet.WordNet("http://wordnet")
    monkeypatch.setattr(wn, "parents", lambda sense: parents[sense])
    assert wn.matches("omw-en-15165490-n")


def test_unknown_sense_is_not_an_outage(monkeypatch):
    def fetch(url, timeout):
        raise HTTPError(url, 404, "not found", {}, None)

    monkeypatch.setattr(wordnet, "urlopen", fetch)
    assert not wordnet.WordNet("http://wordnet").matches("wikidata-en-L1-S1")


def test_ids_cannot_replace_the_configured_host(monkeypatch):
    urls = []

    def fetch(url, timeout):
        urls.append(url)
        return io.BytesIO(b'{"data":{"relationships":{}}}')

    monkeypatch.setattr(wordnet, "urlopen", fetch)
    assert not wordnet.WordNet("http://wordnet").matches("https://other/?secret")
    assert urls == ["http://wordnet/lexicons/omw-en:1.4/synsets/https%3A%2F%2Fother%2F%3Fsecret"]


def test_shared_deadline_bounds_multiple_traversals(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(wordnet, "monotonic", lambda: clock[0])
    wn = wordnet.WordNet("http://wordnet")

    def parents(synset):
        clock[0] += 6
        return ("omw-en-15113229-n",)

    monkeypatch.setattr(wn, "parents", parents)
    assert wn.matches("first", deadline=10)
    with pytest.raises(wordnet.WordNetUnavailableError, match="time budget"):
        wn.matches("second", deadline=10)


@pytest.mark.parametrize("sense", ["omw-en-15113229-n", "omw-en-01825237-v"])
def test_missing_pinned_resource_fails(monkeypatch, sense):
    def fetch(url, timeout):
        raise HTTPError(url, 404, "not found", {}, None)

    monkeypatch.setattr(wordnet, "urlopen", fetch)
    with pytest.raises(wordnet.WordNetUnavailableError):
        wordnet.WordNet("http://wordnet").parents(sense)


def test_volition_uses_ancestry_and_shares_cache_between_categories(monkeypatch):
    # Verified OMW ancestry: wish -> desire; plan -> intend.
    graph = {"omw-en-01824339-v": ["omw-en-01825237-v"],
             "omw-en-00705227-v": ["omw-en-00708538-v"]}
    calls = []

    def fetch(url, timeout):
        sense = url.rsplit("/", 1)[-1]
        calls.append(sense)
        return io.BytesIO(json.dumps({"data": {"relationships": {
            "hypernym": {"data": [{"id": parent} for parent in graph.get(sense, [])]},
        }}}).encode())

    monkeypatch.setattr(wordnet, "urlopen", fetch)
    wn = wordnet.WordNet("http://wordnet")
    assert wn.matches("omw-en-01824339-v", "volition")
    assert not wn.matches("omw-en-01824339-v", "time")
    assert calls.count("omw-en-01824339-v") == 1
    assert wn.matches("omw-en-00705227-v", "volition")
    assert wn.matches("omw-en-02530167-v", "volition")
    assert not wn.matches("omw-en-02632567-v", "volition")
    with pytest.raises(wordnet.WordNetUnavailableError, match="time budget"):
        wn.matches("omw-en-01824339-v", "volition", deadline=0)
