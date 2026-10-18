"""Construction guards distinguish existence, location and conditional scope."""

from copy import deepcopy

import pytest

from spoken_to_signed.text_to_gloss.senses import senses_to_gloss
from tests.test_senses_rules import parsed


def existential():
    doc = parsed(
        [
            ("There", "there", "PRON", "expl", 1),
            ("is", "be", "VERB", "ROOT", 1),
            ("a", "a", "DET", "det", 3),
            ("key", "key", "NOUN", "attr", 1),
            (".", ".", "PUNCT", "punct", 1),
        ],
        synsets=[{"id": "wikidata-en-L2998-S1", "start_token": 0, "end_token": 0}],
    )
    doc["tokens"][1]["morph"] = {"Tense": "Pres", "VerbForm": "Fin"}
    return doc


def test_present_existence_keeps_source_and_target_meanings_separate():
    doc = existential()
    original = deepcopy(doc)
    result = senses_to_gloss(doc)
    assert [i["gloss"] for i in result["sentences"][0]] == ["EXIST", "key", "."]
    assert result["indexes"] == [[0, 3, 4]]
    item = result["sentences"][0][0]
    assert item["synsets"][0]["id"] == "omw-en-02603699-v"
    assert item["source"]["tokens"] == doc["tokens"][:2]
    assert item["rewrite"]["source_candidate_indexes"] == [0, 1]
    assert doc == original


@pytest.mark.parametrize("contrast", ["locative", "unknown", "past", "negative", "question", "modal", "no-flour"])
def test_existence_contrasts_abstain(contrast):
    doc = existential()
    if contrast == "locative":
        doc["synsets"][0]["id"] = "omw-en-00109151-r"
    elif contrast == "unknown":
        doc["synsets"] = []
    elif contrast == "past":
        doc["tokens"][1].update(word="was", morph={"Tense": "Past", "VerbForm": "Fin"})
    elif contrast == "question":
        doc["tokens"][-1]["word"] = "?"
    elif contrast == "no-flour":
        doc["tokens"][2].update(word="no", lemma="no")
    else:
        doc["tokens"][2].update(
            word="not" if contrast == "negative" else "may",
            lemma="not" if contrast == "negative" else "may",
            pos="PART" if contrast == "negative" else "AUX",
            dep="neg" if contrast == "negative" else "aux",
            head=1,
        )
    result = senses_to_gloss(doc)
    assert not any(c["rule"] == "rewrite:present-existential" for c in result["changes"])


def conditional():
    return parsed(
        [
            ("We", "we", "PRON", "nsubj", 1),
            ("leave", "leave", "VERB", "ROOT", 1),
            ("if", "if", "SCONJ", "mark", 4),
            ("Kim", "Kim", "PROPN", "nsubj", 4),
            ("agrees", "agree", "VERB", "advcl", 1),
            (".", ".", "PUNCT", "punct", 1),
        ],
        synsets=[{"id": "wikidata-en-L3003-S1", "start_token": 2, "end_token": 2}],
        entities=[{"id": "person-kim", "start_token": 3, "end_token": 3}],
    )


def test_trailing_condition_moves_as_a_whole_without_splitting_entity():
    doc = conditional()
    original = deepcopy(doc)
    result = senses_to_gloss(doc)
    assert [i["gloss"] for i in result["sentences"][0]] == ["if", "Kim", "agree", "we", "leave", "."]
    assert result["indexes"] == [[2, 3, 4, 0, 1, 5]]
    assert result["changes"] == [{"sentence": 0, "rule": "conditional-frame-first", "source_tokens": [2, 3, 4]}]
    assert doc == original


@pytest.mark.parametrize("contrast", ["embedded", "wrong-sense", "missing-sense", "question", "atomic-boundary"])
def test_conditional_reordering_preserves_uncertain_scope(contrast):
    doc = conditional()
    if contrast == "embedded":
        doc["tokens"][1].update(word="wonder", lemma="wonder")
        doc["tokens"][4]["dep"] = "ccomp"
    elif contrast == "wrong-sense":
        doc["synsets"][0]["id"] = "wikidata-en-L5223-S1"
    elif contrast == "missing-sense":
        doc["synsets"] = []
    elif contrast == "question":
        doc["tokens"][-1]["word"] = "?"
    else:
        doc["synsets"].append({"id": "cross-boundary-meaning", "start_token": 1, "end_token": 2})
    result = senses_to_gloss(doc)
    assert not any(c["rule"] == "conditional-frame-first" for c in result["changes"])


@pytest.mark.parametrize("scope", ["matrix", "embedded"])
def test_embedded_present_copula_omission_preserves_negation_scope(scope):
    if scope == "matrix":
        rows = [
            ("I", "i", "PRON", "nsubj", 3),
            ("do", "do", "AUX", "aux", 3),
            ("not", "not", "PART", "neg", 3),
            ("think", "think", "VERB", "ROOT", 3),
            ("Lee", "Lee", "PROPN", "nsubj", 5),
            ("is", "be", "AUX", "ccomp", 3),
            ("ready", "ready", "ADJ", "acomp", 5),
        ]
        copula, indexes, negative = 5, [0, 2, 3, 4, 6], 1
    else:
        rows = [
            ("I", "i", "PRON", "nsubj", 1),
            ("think", "think", "VERB", "ROOT", 1),
            ("Lee", "Lee", "PROPN", "nsubj", 3),
            ("is", "be", "AUX", "ccomp", 1),
            ("not", "not", "PART", "neg", 3),
            ("ready", "ready", "ADJ", "acomp", 3),
        ]
        copula, indexes, negative = 3, [0, 1, 2, 4, 5], 3
    doc = parsed(rows, synsets=[{"id": "omw-en-02604760-v", "start_token": copula, "end_token": copula}])
    result = senses_to_gloss(doc)
    assert result["indexes"] == [indexes]
    item = result["sentences"][0][negative]
    assert item["gloss"] == "not"
    assert item["source"]["tokens"][0]["head"] == (3 if scope == "matrix" else copula)


@pytest.mark.parametrize("contrast", ["unknown", "wrong-sense", "past", "progressive", "passive", "ellipsis"])
def test_embedded_be_other_functions_remain(contrast):
    doc = parsed(
        [
            ("I", "i", "PRON", "nsubj", 1),
            ("think", "think", "VERB", "ROOT", 1),
            ("Lee", "Lee", "PROPN", "nsubj", 3),
            ("is", "be", "AUX", "ccomp", 1),
            ("ready", "ready", "ADJ", "acomp", 3),
        ],
        synsets=[{"id": "omw-en-02604760-v", "start_token": 3, "end_token": 3}],
    )
    if contrast == "unknown":
        doc["synsets"] = []
    elif contrast == "wrong-sense":
        doc["synsets"][0]["id"] = "omw-en-02603699-v"
    elif contrast == "past":
        doc["tokens"][3].update(word="was", morph={"Tense": "Past"})
    elif contrast == "ellipsis":
        doc["tokens"].pop()
        doc["sentences"][0]["end_token"] = 3
    else:
        doc["tokens"][2].update(head=4, dep="nsubj" if contrast == "progressive" else "nsubjpass")
        doc["tokens"][3].update(dep="aux" if contrast == "progressive" else "auxpass", head=4)
        doc["tokens"][4].update(word="working", lemma="work", pos="VERB", dep="ccomp", head=1)
    result = senses_to_gloss(doc)
    assert any(i["start_token"] == 3 for i in result["sentences"][0])
