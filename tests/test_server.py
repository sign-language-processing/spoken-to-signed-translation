from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from spoken_to_signed import server


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "MODEL_VERSION", "test-build")
    with TestClient(server.app) as client:
        yield client


def request(tokens, **kwargs):
    return {"tokens": tokens, "spoken_language": "en", "signed_language": "ase", **kwargs}


def test_health(client):
    response = client.get("/health")
    assert response.json() == {"status": "healthy", "version": "test-build"}
    assert response.headers["X-Model-Tag"] == "test-build"


def test_question(client):
    tokens = [
        {"word": word, "gloss": gloss, "pos": pos}
        for word, gloss, pos in [
            ("What", "what", "PRON"),
            ("is", "be", "AUX"),
            ("your", "your", "PRON"),
            ("name", "name", "NOUN"),
            ("?", "?", "PUNCT"),
        ]
    ]
    response = client.post("/tokens-to-gloss", json=request(tokens))
    assert response.status_code == 200
    assert response.json() == {"sentences": [[2, 3, 0, 4]]}
    assert response.headers["X-Model-Tag"] == "test-build"


def test_atomic_spans_duplicates_and_unknown_words(client):
    tokens = [
        {"word": "New York", "gloss": "New York", "pos": "PROPN"},
        {"word": "Amit", "gloss": "Amit", "pos": "PROPN"},
        {"word": "Amit", "gloss": "Amit", "pos": "PROPN"},
    ]
    response = client.post("/tokens-to-gloss", json=request(tokens))
    assert response.json() == {"sentences": [[0, 1, 2]]}


def test_morphology_and_sentence_boundaries(client):
    tokens = [
        {"word": "It", "gloss": "it", "pos": "PRON"},
        {"word": "'s", "gloss": "be", "pos": "AUX", "morphology": [{"Tense": "Pres"}]},
        {"word": "fine", "gloss": "fine", "pos": "ADJ"},
        {"word": ".", "gloss": ".", "pos": "PUNCT"},
        {"word": "Go", "gloss": "go", "pos": "VERB"},
        {"word": "!", "gloss": "!", "pos": "PUNCT"},
    ]
    response = client.post("/tokens-to-gloss", json=request(tokens))
    assert response.json() == {"sentences": [[0, 2, 3], [4, 5]]}


def test_simple_and_empty(client):
    response = client.post("/tokens-to-gloss", json=request([], glosser="simple"))
    assert response.json() == {"sentences": [[]]}
    response = client.post("/tokens-to-gloss", json=request([{"gloss": "the"}], glosser="simple"))
    assert response.json() == {"sentences": [[0]]}


@pytest.mark.parametrize(
    "payload",
    [
        request([], spoken_language="fr"),
        request([], signed_language="bfi"),
        request([], glosser="unknown"),
        request([{"word": "missing gloss"}]),
        request([{"gloss": "book", "morphology": "plural"}]),
        request([], language="en"),
    ],
)
def test_invalid_requests(client, payload):
    assert client.post("/tokens-to-gloss", json=payload).status_code == 422


def test_pose_backend_is_optional(client, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("LEXICON_PATH", raising=False)
    assert client.get("/health").status_code == 200
    response = client.post("/gloss-to-pose", json=request([{"gloss": "hello"}]))
    assert response.status_code == 503


def test_pose_endpoint_composes_existing_library(client, monkeypatch):
    import spoken_to_signed.gloss_to_pose as poses

    lookup = MagicMock()
    factory = MagicMock(return_value=lookup)
    monkeypatch.setattr(server, "pose_lookup", factory)
    result = MagicMock()
    result.pose.write.side_effect = lambda buffer: buffer.write(b"pose bytes")
    construct = MagicMock(return_value=result)
    monkeypatch.setattr(poses, "gloss_to_pose", construct)
    response = client.post(
        "/gloss-to-pose",
        json=request(
            [{"gloss": "hello"}, {"gloss": ".", "pos": "PUNCT"}], fingerspelling=False, source="test", anonymize=True
        ),
    )
    assert response.status_code == 200
    assert response.content == b"pose bytes"
    assert response.headers["content-type"] == "application/pose"
    assert response.headers["X-Model-Tag"] == "test-build"
    factory.assert_called_once_with(False, "test")
    assert construct.call_args.args == ([server.GlossItem("hello", "hello")], lookup, "en", "ase")
    assert construct.call_args.kwargs == {"source": "test", "anonymize": True}


def test_pose_request_rejects_empty_content(client, monkeypatch):
    monkeypatch.setattr(server, "pose_lookup", MagicMock())
    assert client.post("/gloss-to-pose", json=request([])).status_code == 422
    assert client.post("/gloss-to-pose", json=request([{"gloss": ".", "pos": "PUNCT"}])).status_code == 422


def test_pose_settings_are_request_local(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
    with_fallback = server.pose_lookup(True, None)
    without_fallback = server.pose_lookup(False, None)
    assert with_fallback is not without_fallback
    assert with_fallback.backup is not None
    assert without_fallback.backup is None
