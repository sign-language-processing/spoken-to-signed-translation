"""Phrase rules must preserve scope, atomic meanings, and source alignment."""

from copy import deepcopy
from unittest.mock import Mock

import pytest

from spoken_to_signed.text_to_gloss.senses import senses_to_gloss
from spoken_to_signed.text_to_gloss.wordnet import WordNet
from tests.test_senses_rules import parsed


def words(document, semantics=None):
    return [item["word"] for item in senses_to_gloss(document, semantics=semantics)["sentences"][0]]


def infinitive(governor="want", sense="omw-en-01825237-v"):
    return parsed([
        ("I", "i", "PRON", "nsubj", 1),
        (governor, governor, "VERB", "ROOT", 1),
        ("to", "to", "PART", "aux", 3),
        ("sleep", "sleep", "VERB", "xcomp", 1),
    ], synsets=[{"id": sense, "start_token": 1, "end_token": 1},
                {"id": "wikidata-en-L2985-S1", "start_token": 2, "end_token": 2}])


@pytest.mark.parametrize("governor", ["want", "wish", "desire", "unseen-word"])
def test_infinitival_complement_uses_meaning_not_spelling(governor):
    doc = infinitive(governor)
    original = deepcopy(doc)
    semantics = Mock(return_value=True)
    assert words(doc, semantics) == ["I", governor, "sleep"]
    semantics.assert_called_once_with("omw-en-01825237-v", "volition")
    assert doc == original


@pytest.mark.parametrize("governor", ["have", "use", "remember"])
def test_unresolved_modality_aspect_and_retrospective_complements(governor):
    assert words(infinitive(governor)) == ["I", governor, "to", "sleep"]


@pytest.mark.parametrize("protection", [
    "missing", "ambiguous", "non-volitional", "marker-sense", "atomic-governor", "entity-governor",
])
def test_infinitive_requires_selected_marker_and_governor_senses(protection):
    doc = infinitive()
    semantics = Mock(return_value=True)
    if protection == "missing":
        doc["synsets"].pop(0)
    elif protection == "ambiguous":
        doc["synsets"].append({**doc["synsets"][0], "id": "other"})
    elif protection == "non-volitional":
        semantics.return_value = False
    elif protection == "marker-sense":
        doc["synsets"][1]["id"] = "directional-to"
    elif protection == "entity-governor":
        doc["entities"] = [{"id": "Q1", "start_token": 1, "end_token": 1}]
    else:
        doc["synsets"][0]["start_token"] = 0
    assert "to" in words(doc, semantics)
    if protection != "non-volitional":
        semantics.assert_not_called()


def test_same_lemma_different_selected_sense(monkeypatch):
    semantics = WordNet("http://wordnet")
    monkeypatch.setattr(semantics, "parents", lambda _: ())
    assert words(infinitive("want", "omw-en-01825237-v"), semantics.matches) == ["I", "want", "sleep"]
    assert words(infinitive("want", "omw-en-02632567-v"), semantics.matches) == ["I", "want", "to", "sleep"]


@pytest.mark.parametrize("protection", ["entity", "meaning", "preposition", "ellipsis"])
def test_to_omission_requires_unprotected_infinitival_syntax(protection):
    doc = infinitive()
    if protection in {"entity", "meaning"}:
        key = "entities" if protection == "entity" else "synsets"
        doc[key] = [{"id": "protected", "start_token": 2, "end_token": 3}]
        assert words(doc, Mock(return_value=True)) == ["I", "want", "to sleep"]
    else:
        doc["tokens"][2].update(pos="ADP", dep="prep", head=1)
        if protection == "preposition":
            doc["tokens"][3].update(word="school", pos="NOUN", dep="pobj", head=2)
        assert "to" in words(doc, Mock(return_value=True))


def temporal(preposition="on"):
    doc = parsed([
        ("We", "we", "PRON", "nsubj", 1),
        ("meet", "meet", "VERB", "ROOT", 1),
        (preposition, preposition, "ADP", "prep", 1),
        ("Monday", "monday", "PROPN", "pobj", 2),
    ], synsets=[{"id": "time", "start_token": 3, "end_token": 3}])
    doc["tokens"][3]["ent_type"] = "DATE"
    return doc


@pytest.mark.parametrize("preposition", ["on", "at"])
def test_temporal_noun_does_not_prove_temporal_pp_role(preposition):
    doc = temporal(preposition)
    assert words(doc, lambda _: True) == ["We", "meet", preposition, "Monday"]


@pytest.mark.parametrize("preposition", ["for", "in", "since", "until", "before", "after"])
def test_other_temporal_relations_are_not_rewritten(preposition):
    assert words(temporal(preposition), lambda _: True) == ["We", "meet", preposition, "Monday"]


@pytest.mark.parametrize("protection", ["no-semantics", "non-time", "idiom", "noun-attachment", "atomic", "argument"])
def test_temporal_pp_guards(protection):
    doc = temporal()
    semantics = Mock(return_value=True)
    if protection == "no-semantics":
        semantics = None
    elif protection == "non-time":
        semantics = Mock(return_value=False)
    elif protection == "idiom":
        doc["tokens"][3].update(word="time", ent_type="")
    elif protection == "noun-attachment":
        doc["tokens"][2]["head"] = 0
    elif protection == "argument":
        doc["tokens"][1].update(word="reflect", lemma="reflect")
    else:
        doc["synsets"].append({"id": "meaning", "start_token": 1, "end_token": 3})
    result = senses_to_gloss(doc, semantics=semantics)
    assert not any(change["rule"] == "temporal-frame-first" for change in result["changes"])


def test_ago_moves_with_its_quantity_and_preserves_direction():
    doc = parsed([
        ("I", "i", "PRON", "nsubj", 1), ("left", "leave", "VERB", "ROOT", 1),
        ("three", "three", "NUM", "nummod", 3), ("days", "day", "NOUN", "npadvmod", 4),
        ("ago", "ago", "ADV", "advmod", 1),
    ], synsets=[{"id": "time", "start_token": 3, "end_token": 3}])
    assert words(doc, lambda _: True) == ["three", "days", "ago", "I", "left"]


def location():
    return parsed([
        ("I", "i", "PRON", "nsubj", 1), ("deposited", "deposit", "VERB", "ROOT", 1),
        ("my", "my", "PRON", "poss", 3), ("paycheck", "paycheck", "NOUN", "dobj", 1),
        ("at", "at", "ADP", "prep", 1), ("the", "the", "DET", "det", 6),
        ("bank", "bank", "NOUN", "pobj", 4), ("yesterday", "yesterday", "NOUN", "npadvmod", 1),
    ], synsets=[
        {"id": "wikidata-en-L3263-S2", "start_token": 4, "end_token": 4},
        {"id": "time", "start_token": 7, "end_token": 7},
    ])


def test_location_then_time_frames_preserve_all_content_and_possession():
    doc = location()
    original = deepcopy(doc)
    result = senses_to_gloss(doc, semantics=lambda sense: sense == "time")
    assert [item["word"] for item in result["sentences"][0]] == [
        "yesterday", "bank", "I", "deposited", "my", "paycheck",
    ]
    assert result["indexes"] == [[7, 6, 0, 1, 2, 3]]
    assert doc == original
    assert any(change["rule"] == "event-location-frame" for change in result["changes"])
    assert {"sentence": 0, "rule": "omit-event-location-marker", "source_tokens": [4]} in result["changes"]


def test_named_location_moves_atomically_with_its_entity_and_senses():
    doc = location()
    doc["tokens"][5].update(word="Central", lemma="central", pos="PROPN", dep="compound", ent_type="FAC")
    doc["tokens"][6].update(word="Park", lemma="park", pos="PROPN", ent_type="FAC")
    doc["entities"] = [{"id": "Q160409", "start_token": 5, "end_token": 6}]
    original = deepcopy(doc)
    result = senses_to_gloss(doc)
    first = result["sentences"][0][0]
    assert first["word"] == "Central Park"
    assert first["entities"] == doc["entities"]
    assert first["source"]["tokens"] == doc["tokens"][5:7]
    assert doc == original


@pytest.mark.parametrize("protection", [
    "unknown-sense", "target-sense", "multiple-senses", "entity", "atomic", "negation", "modal",
    "focus", "noun-attachment", "intransitive", "question", "multiple-prepositions",
])
def test_location_abstains_when_role_or_scope_is_uncertain(protection):
    doc = location()
    if protection == "unknown-sense":
        doc["synsets"] = []
    elif protection == "target-sense":
        doc["synsets"][0]["id"] = "different-at-sense"
    elif protection == "multiple-senses":
        doc["synsets"].append({**doc["synsets"][0], "id": "other"})
    elif protection == "entity":
        doc["entities"] = [{"id": "Q1", "start_token": 4, "end_token": 6}]
    elif protection == "atomic":
        doc["synsets"][0]["end_token"] = 6
    elif protection in {"negation", "modal", "focus", "question", "multiple-prepositions"}:
        word, pos, dep, head = {
            "negation": ("not", "PART", "neg", 1),
            "modal": ("might", "AUX", "aux", 1),
            "focus": ("only", "ADV", "advmod", 4),
            "question": ("?", "PUNCT", "punct", 1),
            "multiple-prepositions": ("near", "ADP", "prep", 1),
        }[protection]
        doc["tokens"][7].update(word=word, lemma=word, pos=pos, dep=dep, head=head)
    elif protection == "noun-attachment":
        doc["tokens"][4]["head"] = 3
    else:
        doc["tokens"][3]["dep"] = "npadvmod"
    result = senses_to_gloss(doc)
    assert not any(change["rule"] == "event-location-frame" for change in result["changes"])
    assert any("at" in item["word"].split() for item in result["sentences"][0])
