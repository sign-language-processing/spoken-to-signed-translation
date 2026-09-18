import os
import subprocess
import sys
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


def test_version_distinguishes_model_overrides_without_credentials():
    def version(**env):
        return subprocess.check_output(
            [sys.executable, "-c", "from spoken_to_signed.server import MODEL_VERSION; print(MODEL_VERSION)"],
            env={**os.environ, "MODEL_VERSION": "build", "OPENAI_MODEL": "gpt-5.6-luna", "OPENAI_BASE_URL": "", **env},
            text=True,
        ).strip()

    baseline = version()
    assert baseline == version(OPENAI_API_KEY="not-a-real-key")
    assert baseline != version(OPENAI_MODEL="qwen3.5-0.8b")
    assert baseline != version(OPENAI_BASE_URL="http://localhost:1234/v1")
    assert version(MODEL_VERSION="") == ""


@pytest.mark.parametrize("glosser", ["rules", "gpt"])
def test_question(client, monkeypatch, glosser):
    if glosser == "gpt":
        from spoken_to_signed.text_to_gloss import gpt

        upstream = MagicMock()
        upstream.chat.completions.create.return_value.choices[0].message.content = '[2, 3, 0, 4]'
        monkeypatch.setattr(gpt, "get_openai_client", lambda: upstream)
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
    tokens[1]["synsets"] = [{"id": "be.v.01"}]
    tokens[3].update(synsets=[{"id": "name.n.01", "confidence": 0.9}], start_token=3, end_token=3)
    response = client.post("/tokens-to-gloss", json=request(tokens, glosser=glosser))
    assert response.status_code == 200
    assert response.json() == {"sentences": [[tokens[2], tokens[3], tokens[0], tokens[4]]], "indexes": [[2, 3, 0, 4]]}
    assert response.headers["X-Model-Tag"] == "test-build"


def test_atomic_spans_duplicates_and_unknown_words(client):
    tokens = [
        {"word": "New York", "gloss": "New York", "pos": "PROPN", "start_token": 0, "end_token": 1},
        {"word": "Amit", "gloss": "Amit", "pos": "PROPN", "start_token": 2},
        {"word": "Amit", "gloss": "Amit", "pos": "PROPN", "start_token": 3},
    ]
    response = client.post("/tokens-to-gloss", json=request(tokens))
    assert response.json() == {"sentences": [tokens], "indexes": [[0, 1, 2]]}


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
    assert response.json() == {
        "sentences": [[tokens[0], tokens[2], tokens[3]], tokens[4:]],
        "indexes": [[0, 2, 3], [4, 5]],
    }


def test_simple_and_empty(client):
    response = client.post("/tokens-to-gloss", json=request([], glosser="simple"))
    assert response.json() == {"sentences": [[]], "indexes": [[]]}
    response = client.post("/tokens-to-gloss", json=request([{"gloss": "the"}], glosser="simple"))
    assert response.json() == {"sentences": [[{"gloss": "the"}]], "indexes": [[0]]}


@pytest.mark.parametrize("glosser", ["rules", "simple"])
def test_annotations_round_trip_without_adding_defaults(client, glosser):
    tokens = [
        {"gloss": "minimal"},
        {
            "word": "books",
            "gloss": "book",
            "pos": "NOUN",
            "morphology": [{"Number": "Plur"}],
            "synsets": [{"id": "book.n.01", "confidence": 0.9}],
            "entities": [],
            "span": {"start_token": 1, "end_token": 1, "start_char": 8, "end_char": 13},
            "annotation": None,
        },
        {"word": None, "gloss": "explicit defaults", "pos": None, "morphology": []},
    ]
    response = client.post("/tokens-to-gloss", json=request(tokens, glosser=glosser))
    assert response.status_code == 200
    assert response.json() == {"sentences": [tokens], "indexes": [[0, 1, 2]]}


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
            [{"gloss": "hello", "synsets": [{"id": "hello.n.01"}]}, {"gloss": ".", "pos": "PUNCT"}],
            fingerspelling=False,
            source="test",
            anonymize=True,
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
