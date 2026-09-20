import json
from unittest.mock import Mock

import httpx
import pytest
from pose_format import Pose

from spoken_to_signed.gloss_to_media import Dictionary, LookupUnavailableError, MissingSignError, Realizer, spell

PATH = "gs://dictionary-videos/original.mp4"


def token(word="Ada", **kwargs):
    return {"word": word, "gloss": word.lower(), "synsets": [{"id": "wikidata-en-L1-S1"}],
            "entities": [{"id": 42}], **kwargs}


def row(identifier=1, *, entity=None, sense=None, path=PATH, fsw="FSW", confidence=1):
    return {"id": identifier, "confidence": confidence, "wikidata_id": entity, "wordnet_synset_id": sense,
            "bucket_url": path, "signwriting": fsw}


def dictionary(rows, calls=None):
    def respond(request):
        if calls is not None:
            calls.append(request)
        return httpx.Response(200, json={"success": True, "data": rows})
    return Dictionary(httpx.Client(transport=httpx.MockTransport(respond)), "http://dictionary")


def test_mixed_batch_is_one_request_and_preserves_entity_then_sense_ranking():
    calls = []
    lookup = dictionary([row(2, entity="Q42", confidence=0.4), row(1, entity="Q42", confidence=0.4),
                         row(3, sense="wikidata-en-L1-S1")], calls)
    hits = lookup.candidates([token()] * 200, "ase")
    assert [[hit.id for hit in group] for group in hits] == [[1, 2, 3]] * 200
    assert len(calls) == 1
    assert calls[0].method == "POST"
    assert calls[0].url.path == "/internal/links"
    assert json.loads(calls[0].content) == {
        "wikidata_ids": ["Q42"], "wordnet_synset_ids": ["wikidata-en-L1-S1"], "signed_language": "ase",
    }


def test_distinct_concepts_are_not_split_into_many_requests():
    calls = []
    dictionary([], calls).candidates([token(entities=[], synsets=[{"id": f"s{i}"}]) for i in range(200)], "ase")
    assert len(calls) == 1
    assert len(json.loads(calls[0].content)["wordnet_synset_ids"]) == 200
    with pytest.raises(ValueError, match="1024"):
        dictionary([], calls).candidates([token(entities=[], synsets=[{"id": f"s{i}"}]) for i in range(1025)], "ase")
    assert len(calls) == 1


def test_entity_without_video_uses_sense_and_miss_retains_original_text():
    realizer = Realizer(dictionary([row(entity="Q42", path=None, fsw=None),
                                   row(2, sense="wikidata-en-L1-S1")]))
    items = [token(), token("Amit", entities=[], synsets=[])]
    assert realizer.resolve(items, "video", "en", "ase") == [{"bucket_url": PATH}, {"text": "Amit"}]
    assert realizer.resolve(items[:1], "signwriting", "en", "ase") == ["FSW"]


@pytest.mark.parametrize("body", [[], {}, {"success": False, "data": []},
                                 {"success": True, "data": [{"id": 1}]},
                                 {"success": True, "data": [row(entity="Q999")]}])
def test_invalid_dictionary_response_is_not_a_miss(body):
    lookup = Dictionary(httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=body))),
                        "http://dictionary")
    with pytest.raises(LookupUnavailableError):
        Realizer(lookup).resolve([token()], "video", "en", "ase")


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
def test_dictionary_outages_never_become_missing_videos(status):
    lookup = Dictionary(httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(status))),
                        "http://dictionary")
    with pytest.raises(LookupUnavailableError):
        Realizer(lookup).resolve([token()], "video", "en", "ase")


def test_fingerspelling_is_deferred_until_pose_conversion():
    from spoken_to_signed.gloss_to_pose.concatenate import ConcatenationSettings, concatenate_poses

    original = ConcatenationSettings.is_reduce_holistic
    assert Realizer(dictionary([])).resolve([token("Amit🙂")], "video", "en", "ase") == [{"text": "Amit🙂"}]
    pose = Pose.read(spell(token("Amit"), "pose", "en", "ase"))
    assert len(pose.body) > 0
    assert len(next(c for c in pose.header.components if c.name == "FACE_LANDMARKS").points) == 478
    assert ConcatenationSettings.is_reduce_holistic == original
    assert len(concatenate_poses([pose], reduce=False, max_sign_seconds=None).body) > 0
    assert spell(token("Amit🙂"), "pose", "en", "ase") is None


def test_signwriting_fallback_remains_deterministic():
    realizer = Realizer(dictionary([]))
    assert realizer.resolve([token("www")], "signwriting", "en", "ase") == \
        realizer.resolve([token("www")], "signwriting", "en", "ase")
    with pytest.raises(MissingSignError):
        realizer.resolve([token("Amit")], "signwriting", "en", "ase", fingerspelling=False)


def test_contained_senses_are_not_used_for_whole_entities():
    calls = []
    dictionary([], calls).candidates([token("New York", synsets=[], source={"synsets": [{"id": "york"}]})], "ase")
    assert json.loads(calls[0].content)["wordnet_synset_ids"] == []
    calls.clear()
    dictionary([], calls).candidates([token(entities=[], synsets=[])], "ase")
    assert calls == []


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
