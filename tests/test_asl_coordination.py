"""Independent clauses reuse rules without moving meaning across a conjunction."""

from copy import deepcopy

import pytest

from spoken_to_signed.text_to_gloss.senses import senses_to_gloss
from tests.test_senses_rules import parsed


def coordinated():
    return parsed([
        ("I", "i", "PRON", "nsubj", 1), ("bought", "buy", "VERB", "ROOT", 1),
        ("books", "book", "NOUN", "dobj", 1), ("yesterday", "yesterday", "NOUN", "npadvmod", 1),
        (",", ",", "PUNCT", "punct", 1), ("and", "and", "CCONJ", "cc", 1),
        ("she", "she", "PRON", "nsubj", 7), ("sold", "sell", "VERB", "conj", 1),
        ("cars", "car", "NOUN", "dobj", 7), ("today", "today", "NOUN", "npadvmod", 7),
        (".", ".", "PUNCT", "punct", 1),
    ], synsets=[{"id": "time", "start_token": i, "end_token": i} for i in (3, 9)])


def test_time_frames_stay_in_their_clause_with_original_alignment():
    doc = coordinated()
    original = deepcopy(doc)
    result = senses_to_gloss(doc, semantics=lambda _: True)
    assert result["indexes"] == [[3, 0, 1, 2, 5, 9, 6, 7, 8]]
    assert len(result["sentences"]) == 1
    assert doc == original
    assert [c["source_tokens"] for c in result["changes"] if c["rule"] == "temporal-frame-first"] == [[3], [9]]
    assert result["notes"] == []
    for item in result["sentences"][0]:
        assert item["source"]["tokens"] == original["tokens"][item["start_token"]:item["end_token"] + 1]


@pytest.mark.parametrize("guard", [
    "shared-subject", "shared-object", "shared-time", "embedded", "question", "quotation", "correlative",
    "atomic-boundary", "missing-conjunction", "misplaced-conjunction", "subordinate-marker",
])
def test_uncertain_coordination_preserves_order(guard):
    doc = coordinated()
    if guard == "atomic-boundary":
        doc["entities"] = [{"id": "Q1", "start_token": 3, "end_token": 6}]
    else:
        index, update = {
            "shared-subject": (6, {"dep": "dobj", "head": 1}),
            "shared-object": (8, {"head": 1}),
            "shared-time": (9, {"head": 1}),
            "embedded": (7, {"dep": "ccomp"}),
            "question": (10, {"word": "?"}),
            "quotation": (4, {"word": '"'}),
            "correlative": (4, {"word": "either", "pos": "CCONJ", "dep": "preconj"}),
            "missing-conjunction": (5, {"dep": "advmod"}),
            "misplaced-conjunction": (5, {"head": 2}),
            "subordinate-marker": (4, {"word": "if", "pos": "SCONJ", "dep": "mark"}),
        }[guard]
        doc["tokens"][index].update(update)
    result = senses_to_gloss(doc, semantics=lambda _: True)
    assert result["indexes"][0] == sorted(result["indexes"][0])
    assert not any(c["rule"] == "temporal-frame-first" for c in result["changes"])


def test_named_subject_stays_atomic_inside_its_clause():
    doc = coordinated()
    doc["entities"] = [{"id": "Q1", "start_token": 6, "end_token": 6}]
    result = senses_to_gloss(doc, semantics=lambda _: True)
    assert result["indexes"] == [[3, 0, 1, 2, 5, 9, 6, 7, 8]]
    assert result["sentences"][0][6]["entities"] == doc["entities"]
