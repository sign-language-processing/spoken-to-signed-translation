"""Degree and auxiliary contrasts over the actual WSD entry point."""

from copy import deepcopy

import pytest

from spoken_to_signed.text_to_gloss.rules import degree_aware_lemma
from spoken_to_signed.text_to_gloss.senses import senses_to_gloss
from tests.test_senses_rules import parsed


@pytest.mark.parametrize(
    ("word", "lemma", "morph", "expected"),
    [
        ("best", "good", {"Degree": "Sup"}, "best"),
        ("worst", "bad", {"Degree": "Sup"}, "worst"),
        ("better", "good", {"Degree": "Cmp"}, "better"),
        ("happier", "happy", {"Degree": "Cmp"}, "happier"),
        ("longer", "long", {"Degree": "Cmp"}, "longer"),
        ("good", "good", {"Degree": "Pos"}, "good"),
        ("books", "book", {"Number": "Plur"}, "book"),
    ],
)
def test_degree_is_preserved_before_glossing(word, lemma, morph, expected):
    assert degree_aware_lemma(word, lemma, morph) == expected
    doc = parsed([(word, lemma, "ADJ", "ROOT", 0)])
    doc["tokens"][0]["morph"] = morph
    doc["synsets"] = [{"id": "selected-sense", "start_token": 0, "end_token": 0}]
    original = deepcopy(doc)
    item = senses_to_gloss(doc)["sentences"][0][0]
    assert item["gloss"] == expected
    assert item["morphology"] == [morph]
    assert item["synsets"] == doc["synsets"]
    assert doc == original


@pytest.mark.parametrize("auxiliary", ["have", "has", "had", "'ve", "'s", "'d"])
@pytest.mark.parametrize("negative", [False, True])
def test_perfect_have_is_omitted_without_losing_negation_or_participle_features(auxiliary, negative):
    rows = [
        ("She", "she", "PRON", "nsubj", 2),
        (auxiliary, "have", "AUX", "aux", 2),
        ("eaten", "eat", "VERB", "ROOT", 2),
    ]
    if negative:
        rows.append(("not", "not", "PART", "neg", 2))
    doc = parsed(rows)
    doc["tokens"][2]["morph"] = {"VerbForm": "Part", "Aspect": "Perf"}
    # An incorrect lexical WSD HAVE sense must not override clear AUX syntax.
    doc["synsets"] = [{"id": "omw-en-02203362-v", "start_token": 1, "end_token": 1}]
    original = deepcopy(doc)
    result = senses_to_gloss(doc)
    assert result["indexes"] == [[0, 2, 3] if negative else [0, 2]]
    assert result["sentences"][0][1]["morphology"] == [{"VerbForm": "Part", "Aspect": "Perf"}]
    assert any(note["code"] == "perfect-aspect-not-realized" for note in result["notes"])
    assert doc == original


@pytest.mark.parametrize("pos", ["VERB", "AUX"])
def test_nonperfect_have_is_preserved(pos):
    doc = parsed(
        [("I", "i", "PRON", "nsubj", 1), ("have", "have", pos, "ROOT", 1), ("books", "book", "NOUN", "dobj", 1)]
    )
    assert senses_to_gloss(doc)["indexes"] == [[0, 1, 2]]


def test_causative_have_is_not_perfect_have():
    doc = parsed(
        [
            ("I", "i", "PRON", "nsubj", 1),
            ("had", "have", "VERB", "ROOT", 1),
            ("it", "it", "PRON", "nsubj", 3),
            ("repaired", "repair", "VERB", "ccomp", 1),
        ]
    )
    doc["tokens"][3]["morph"] = {"VerbForm": "Part"}
    assert senses_to_gloss(doc)["indexes"] == [[0, 1, 2, 3]]


def test_perfect_expression_is_atomic_even_with_a_have_auxiliary_head():
    doc = parsed(
        [("has", "have", "AUX", "aux", 2), ("just", "just", "ADV", "advmod", 2), ("eaten", "eat", "VERB", "ROOT", 2)]
    )
    doc["tokens"][2]["morph"] = {"VerbForm": "Part"}
    doc["synsets"] = [{"id": "expression", "expression": "has just", "start_token": 0, "end_token": 1}]
    assert senses_to_gloss(doc)["indexes"] == [[0, 1]]
