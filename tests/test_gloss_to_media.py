import base64
from unittest.mock import Mock

import httpx
import pytest
from pose_format import Pose

from spoken_to_signed.gloss_to_media import Dictionary, LookupUnavailableError, MissingSignError, PoseStore, Realizer

MD5 = "a" * 32


def token(word="Ada", **kwargs):
    return {"word": word, "gloss": word.lower(), "synsets": [{"id": "wikidata-en-L1-S1"}],
            "entities": [{"id": 42}], **kwargs}


def row(identifier=1, *, entity=None, sense=None, md5=MD5, fsw="FSW", confidence=1):
    return {"id": identifier, "confidence": confidence, "wikidata_id": entity, "wordnet_synset_id": sense,
            "assets": {"video_md5": md5, "signwriting": fsw}}


def dictionary(respond):
    return Dictionary(httpx.Client(transport=httpx.MockTransport(respond)), "http://dictionary")


def test_batch_deduplicates_concepts_and_keeps_entity_then_sense_order():
    calls = []

    def respond(request):
        calls.append(request)
        assert request.url.path == "/internal/links"
        assert request.url.params["signed_language"] == "ase"
        rows = ([row(2, entity="Q42", confidence=0.4), row(1, entity="Q42", confidence=0.4)]
                if "wikidata_id" in request.url.params else [row(3, sense="wikidata-en-L1-S1")])
        return httpx.Response(200, json={"success": True, "data": rows})

    candidates = dictionary(respond).candidates([token(), token()], "ase")
    assert [[hit.id for hit in hits] for hits in candidates] == [[1, 2, 3], [1, 2, 3]]
    assert len(calls) == 2


def test_dictionary_batches_at_100_identifiers():
    sizes = []

    def respond(request):
        sizes.append(len(request.url.params["wordnet_synset_id"].split(",")))
        return httpx.Response(200, json={"success": True, "data": []})

    dictionary(respond).candidates([token(entities=[], synsets=[{"id": f"s{i}"}]) for i in range(101)], "ase")
    assert sizes == [100, 1]


def test_entity_without_requested_asset_falls_back_to_lexical_sense():
    def respond(request):
        rows = ([row(entity="Q42", md5=None, fsw=None)] if "wikidata_id" in request.url.params
                else [row(2, sense="wikidata-en-L1-S1")])
        return httpx.Response(200, json={"success": True, "data": rows})

    store = Mock(exists=Mock(return_value=True))
    realizer = Realizer(dictionary(respond), store)
    assert realizer.resolve([token(), token()], "pose", "en", "ase") == [{"md5": MD5}] * 2
    store.exists.assert_called_once_with(MD5)
    assert realizer.resolve([token()], "signwriting", "en", "ase") == ["FSW"]


@pytest.mark.parametrize("body", [[], {}, {"success": False, "data": []},
                                 {"success": True, "data": [{"id": 1}]},
                                 {"success": True, "data": [row(entity="Q999")]},
                                 {"success": True, "data": [row(entity="Q42", md5="../escape")]}])
def test_invalid_dictionary_response_is_not_a_miss(body):
    with pytest.raises(LookupUnavailableError):
        Realizer(dictionary(lambda _: httpx.Response(200, json=body))).resolve([token()], "signwriting", "en", "ase")


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
def test_dictionary_outages_never_become_fingerspelling(status):
    with pytest.raises(LookupUnavailableError):
        Realizer(dictionary(lambda _: httpx.Response(status))).resolve([token()], "pose", "en", "ase")


def test_storage_failures_propagate():
    from google.api_core.exceptions import Forbidden

    store = PoseStore("bucket")
    store.bucket = Mock()
    store.bucket.blob.return_value.exists.side_effect = Forbidden("no permission")
    with pytest.raises(LookupUnavailableError):
        store.exists(MD5)


def test_missing_pose_fingerspells_and_preserves_full_holistic_layout():
    from spoken_to_signed.gloss_to_pose.concatenate import ConcatenationSettings, concatenate_poses

    lookup = dictionary(lambda _: httpx.Response(200, json={"success": True, "data": [row(entity="Q42")]}))
    original = ConcatenationSettings.is_reduce_holistic
    result = Realizer(lookup, Mock(exists=Mock(return_value=False))).resolve(
        [token("Amit", synsets=[])], "pose", "en", "ase")
    pose = Pose.read(base64.b64decode(result[0]["base64"]))
    assert len(pose.body) > 0
    assert len(next(c for c in pose.header.components if c.name == "FACE_LANDMARKS").points) == 478
    assert ConcatenationSettings.is_reduce_holistic == original
    joined = concatenate_poses([pose], reduce=False, max_sign_seconds=None)
    assert len(joined.body) > 0


def test_fingerspelling_is_deterministic_and_never_drops_unsupported_characters():
    realizer = Realizer(dictionary(lambda _: httpx.Response(200, json={"success": True, "data": []})))
    assert realizer.resolve([token("www")], "signwriting", "en", "ase") == \
        realizer.resolve([token("www")], "signwriting", "en", "ase")
    for target in ("pose", "signwriting"):
        with pytest.raises(MissingSignError):
            realizer.resolve([token("Amit")], target, "en", "ase", fingerspelling=False)
        with pytest.raises(MissingSignError):
            realizer.resolve([token("Amit🙂")], target, "en", "ase")


def test_contained_senses_are_not_used_as_whole_entity_fallback():
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json={"success": True, "data": []})

    value = token("New York", synsets=[], source={"synsets": [{"id": "york-sense"}]})
    dictionary(respond).candidates([value], "ase")
    assert len(calls) == 1
    assert "wordnet_synset_id" not in calls[0].url.params


def test_internal_lookup_uses_google_identity_unless_local_bypass(monkeypatch):
    from spoken_to_signed import gloss_to_media

    monkeypatch.setenv("DICTIONARY_API_URL", "https://dictionary.example")
    mint = Mock(return_value="identity-token")
    monkeypatch.setattr(gloss_to_media, "identity_token", mint)
    headers = []

    def respond(request):
        assert request.url.path == "/internal/links"
        headers.append(request.headers.get("authorization"))
        return httpx.Response(200, json={"success": True, "data": []})

    original = httpx.Client
    monkeypatch.setattr(gloss_to_media.httpx, "Client",
                        lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs))
    for bypass in ("false", "true"):
        monkeypatch.setenv("SKIP_AUTH", bypass)
        gloss_to_media.realize([token("Amit", entities=[])], "signwriting", "en", "ase")
    assert headers == ["Bearer identity-token", None]
    assert mint.call_count == 1
    assert mint.call_args.args[0] == "https://dictionary.example"


def test_identity_credentials_reuse_and_refresh(monkeypatch):
    from spoken_to_signed import gloss_to_media

    credentials = Mock(valid=True, token="cached")
    monkeypatch.setattr(gloss_to_media, "_identity_credentials", Mock(return_value=credentials))
    assert gloss_to_media.identity_token("https://dictionary") == "cached"
    credentials.refresh.assert_not_called()
    credentials.valid = False
    credentials.refresh.side_effect = lambda _: setattr(credentials, "token", "refreshed")
    assert gloss_to_media.identity_token("https://dictionary") == "refreshed"
    credentials.refresh.assert_called_once()
