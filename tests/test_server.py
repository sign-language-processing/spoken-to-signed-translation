from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from spoken_to_signed import server


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "MODEL_VERSION", "test-build")
    monkeypatch.setattr(server, "semantics", None)
    with TestClient(server.app) as client:
        yield client


def request(tokens, **kwargs):
    return {"tokens": tokens, "spoken_language": "en", "signed_language": "ase", **kwargs}


def senses_request(**kwargs):
    return {
        "spoken_language": "en",
        "signed_language": "ase",
        "senses": {
            "tokens": [{"word": "hello", "lemma": "hello", "pos": "INTJ", "dep": "ROOT", "head": 0}],
            "synsets": [],
            "entities": [],
            "sentences": [{"start_token": 0, "end_token": 0}],
        },
        **kwargs,
    }


def test_health(client):
    response = client.get("/health")
    assert response.json() == {"status": "healthy", "version": "test-build"}
    assert response.headers["X-Model-Tag"] == "test-build"


def test_senses_endpoint_needs_no_model(client):
    response = client.post("/senses-to-gloss", json=senses_request())
    assert response.status_code == 200
    assert response.json()["indexes"] == [[0]]
    assert response.json()["sentences"][0][0]["word"] == "hello"
    assert response.headers["X-Model-Tag"] == "test-build"
    assert response.json()["notes"] == [{"sentence": 0, "code": "temporal-semantics-unavailable"}]


def test_retired_endpoint(client):
    assert client.post("/tokens-to-gloss", json=request([])).status_code == 404


@pytest.mark.parametrize(
    "options",
    [
        {"spoken_language": "fr"},
        {"signed_language": "bfi"},
        {"glosser": "gpt"},
        {"language": "en"},
        {"senses": {"tokens": [], "entities": [], "synsets": []}},
    ],
)
def test_invalid_requests(client, options):
    assert client.post("/senses-to-gloss", json=senses_request(**options)).status_code == 422


def test_bad_dependency_is_422(client):
    body = senses_request()
    body["senses"]["tokens"][0]["head"] = 1
    assert client.post("/senses-to-gloss", json=body).status_code == 422


def test_wordnet_outage_is_not_success(client, monkeypatch):
    from spoken_to_signed.text_to_gloss.wordnet import WordNetUnavailableError

    monkeypatch.setattr(server, "gloss_senses", MagicMock(side_effect=WordNetUnavailableError("unavailable")))
    response = client.post("/senses-to-gloss", json=senses_request())
    assert response.status_code == 503
    assert response.json() == {"detail": "unavailable"}


def test_pose_backend_is_optional(client, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("LEXICON_PATH", raising=False)
    assert client.get("/health").status_code == 200
    assert client.post("/gloss-to-pose", json=request([{"gloss": "hello"}])).status_code == 503


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
