"""Lexical lowering must preserve comparison boundaries, arguments and negation."""

from copy import deepcopy

from spoken_to_signed.text_to_gloss.senses import senses_to_gloss
from tests.test_senses_rules import parsed


def test_tolerate_consumes_only_its_licensed_with_complement():
    doc = parsed(
        [
            ("I", "i", "PRON", "nsubj", 1),
            ("put", "put", "VERB", "ROOT", 1),
            ("up", "up", "ADP", "prt", 1),
            ("with", "with", "ADP", "prep", 1),
            ("noise", "noise", "NOUN", "pobj", 3),
        ],
        synsets=[{"id": "omw-en-00668099-v", "expression": "put up", "start_token": 1, "end_token": 2}],
    )
    original = deepcopy(doc)
    result = senses_to_gloss(doc)
    assert [i["gloss"] for i in result["sentences"][0]] == ["i", "TOLERATE", "noise"]
    assert result["indexes"] == [[0, 1, 3]]
    item = result["sentences"][0][1]
    assert item["rewrite"]["source_candidate_indexes"] == [1, 2]
    assert item["source"]["tokens"] == doc["tokens"][1:4]
    assert doc == original


def test_tolerating_with_patience_does_not_consume_instrumental_with():
    doc = parsed(
        [
            ("I", "i", "PRON", "nsubj", 1),
            ("bear", "bear", "VERB", "ROOT", 1),
            ("with", "with", "ADP", "prep", 1),
            ("patience", "patience", "NOUN", "pobj", 2),
        ],
        synsets=[{"id": "omw-en-00668099-v", "start_token": 1, "end_token": 1}],
    )
    result = senses_to_gloss(doc)
    assert [i["gloss"] for i in result["sentences"][0]] == ["i", "bear", "with", "patience"]
    assert result["changes"] == []


def test_additive_rule_does_not_consume_prefix_of_as_well_as():
    doc = parsed(
        [
            ("She", "she", "PRON", "nsubj", 1),
            ("sings", "sing", "VERB", "ROOT", 1),
            ("as", "as", "SCONJ", "mark", 3),
            ("well", "well", "ADV", "advmod", 1),
            ("as", "as", "ADP", "prep", 3),
            ("Kim", "Kim", "PROPN", "pobj", 4),
        ],
        synsets=[{"id": "omw-en-00047534-r", "expression": "as well", "start_token": 2, "end_token": 3}],
    )
    result = senses_to_gloss(doc)
    assert [i["gloss"] for i in result["sentences"][0]] == ["she", "sing", "as well", "as", "Kim"]
    assert result["changes"] == []


def test_decomposing_unaware_does_not_simplify_an_outer_negation():
    doc = parsed(
        [
            ("I", "i", "PRON", "nsubj", 1),
            ("am", "be", "AUX", "ROOT", 1),
            ("not", "not", "PART", "neg", 1),
            ("unaware", "unaware", "ADJ", "acomp", 1),
            ("of", "of", "ADP", "prep", 3),
            ("danger", "danger", "NOUN", "pobj", 4),
        ],
        synsets=[{"id": "omw-en-00191603-a", "start_token": 3, "end_token": 3}],
    )
    original = deepcopy(doc)
    result = senses_to_gloss(doc)
    assert [i["gloss"] for i in result["sentences"][0]] == ["i", "not", "NOT", "AWARE", "of", "danger"]
    assert result["indexes"] == [[0, 2, 3, 3, 4, 5]]
    assert doc == original
