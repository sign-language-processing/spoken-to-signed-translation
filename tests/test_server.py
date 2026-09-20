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


def test_wordnet_startup_allows_cold_start_without_relaxing_requests(monkeypatch):
    import io

    from spoken_to_signed.text_to_gloss import wordnet

    timeouts = []

    def fetch(url, timeout):
        timeouts.append(timeout)
        if len(timeouts) == 1 and timeout < 10:
            raise TimeoutError("WordNet is cold")
        return io.BytesIO(b'{"data":{"relationships":{}}}')

    monkeypatch.setattr(wordnet, "urlopen", fetch)
    monkeypatch.setattr(server, "semantics", wordnet.WordNet("http://wordnet"))
    with TestClient(server.app) as client:
        assert client.get("/health").status_code == 200
        assert server.semantics.parents("other") == ()
    assert timeouts == [60, 5]


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
    monkeypatch.delenv("DICTIONARY_API_URL", raising=False)
    assert client.get("/health").status_code == 200
    assert client.post("/gloss-to-pose", json=request([{"gloss": "hello"}])).status_code == 503


@pytest.mark.parametrize(("target", "field", "result"), [("pose", "poses", [{"md5": "a" * 32}]),
                                               ("signwriting", "signwriting", ["FSW"])])
def test_media_endpoints_preserve_one_result_per_gloss(client, monkeypatch, target, field, result):
    resolve = MagicMock(return_value=result)
    monkeypatch.setattr(server, "realize", resolve)
    response = client.post(f"/gloss-to-{target}", json=request(
        [{"gloss": "hello", "synsets": [{"id": "hello.n.01"}]}], fingerspelling=False))
    assert response.status_code == 200
    assert response.json() == {field: result}
    assert response.headers["X-Model-Tag"] == "test-build"
    assert response.headers["Cache-Control"] == "no-store"
    assert resolve.call_args.args[0][0]["synsets"] == [{"id": "hello.n.01"}]
    assert resolve.call_args.args[1:] == (target, "en", "ase", False)


def test_media_empty_batch_and_invalid_tokens(client):
    assert client.post("/gloss-to-pose", json=request([])).json() == {"poses": []}
    assert client.post("/gloss-to-pose", json=request([{"gloss": ".", "pos": "PUNCT"}])).status_code == 422
    assert client.post("/gloss-to-pose", json=request([{"gloss": "a" * 129}])).status_code == 422
    assert client.post("/gloss-to-pose", json=request([{"gloss": "a"}], signed_language="../../x")).status_code == 422
    assert client.post("/gloss-to-pose", json=request([{"gloss": "a", "synsets": [{"id": "a,b"}]}])).status_code == 422


@pytest.mark.parametrize(("error", "status"), [(server.LookupUnavailableError("offline"), 503),
                                        (server.MissingSignError("missing"), 404)])
def test_media_outage_and_miss_are_distinct(client, monkeypatch, error, status):
    monkeypatch.setattr(server, "realize", MagicMock(side_effect=error))
    assert client.post("/gloss-to-signwriting", json=request([{"gloss": "a"}])).status_code == status


def test_request_limits(client):
    body = senses_request()
    body["senses"]["tokens"] *= 1025
    assert client.post("/senses-to-gloss", json=body).status_code == 422
    assert client.post("/senses-to-gloss", content=b" " * (2 * 1024 * 1024 + 1)).status_code == 413


def test_sentence_metadata_survives_flattening(client):
    result = client.post("/senses-to-gloss", json=senses_request()).json()
    assert result["sentences"][0][0]["sentence"] == 0
    assert result["sentences"][0][0]["notes"] == ["temporal-semantics-unavailable"]
