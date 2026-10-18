"""Semantic lowering must never guess a meaning from the English spelling."""

from copy import deepcopy

import pytest

from spoken_to_signed.text_to_gloss import rewrites
from spoken_to_signed.text_to_gloss.senses import senses_to_gloss
from tests.test_senses_rules import parsed


def meaning(word, sense, *, morphology=None):
    doc = parsed(
        [(word, word.lower(), "ADJ", "ROOT", 0)],
        synsets=[
            {
                "id": sense,
                "definition": "selected source meaning",
                "start_token": 0,
                "end_token": 0,
            }
        ],
    )
    doc["tokens"][0]["morph"] = morphology or {}
    return doc


@pytest.mark.parametrize(
    "rule", [r for r in rewrites.rules() if len(r["sources"]) == 1 and not r.get("guard")], ids=lambda r: r["id"]
)
def test_same_meaning_rewrites_independently_of_surface_form(rule):
    doc = meaning("unseen-surface", rule["sources"][0]["sense_id"])
    original = deepcopy(doc)
    result = senses_to_gloss(doc)
    items = result["sentences"][0]
    assert [i["gloss"] for i in items] == [t["form"] for t in rule["targets"]]
    assert [i["synsets"][0]["id"] for i in items] == [t["sense_id"] for t in rule["targets"]]
    assert result["indexes"] == [[0] * len(items)]
    assert result["changes"][-1]["rule"] == "rewrite:" + rule["id"]
    assert all(i["source"]["synsets"] == doc["synsets"] for i in items)
    assert all(i["source"]["tokens"] == doc["tokens"] for i in items)
    assert doc == original


@pytest.mark.parametrize("clause_dep", ["ccomp", "csubj", "acl"])
def test_clause_that_omission_requires_object_clause(clause_dep):
    doc = parsed(
        [
            ("We", "we", "PRON", "nsubj", 1),
            ("ask", "ask", "VERB", "ROOT", 1),
            ("that", "that", "SCONJ", "mark", 4),
            ("you", "you", "PRON", "nsubj", 4),
            ("stay", "stay", "VERB", clause_dep, 1),
        ],
        synsets=[{"id": "wikidata-en-L2988-S1", "start_token": 2, "end_token": 2}],
    )
    original = deepcopy(doc)
    result = senses_to_gloss(doc)
    assert [i["gloss"] for i in result["sentences"][0]] == (
        ["we", "ask", "you", "stay"] if clause_dep == "ccomp" else ["we", "ask", "that", "you", "stay"]
    )
    assert result["indexes"] == ([[0, 1, 3, 4]] if clause_dep == "ccomp" else [[0, 1, 2, 3, 4]])
    assert doc == original


@pytest.mark.parametrize("head_pos", ["ADJ", "ADV", "VERB"])
def test_very_canonicalization_requires_degree_modifier(head_pos):
    doc = parsed(
        [("really", "really", "ADV", "advmod", 1), ("hot", "hot", head_pos, "ROOT", 1)],
        synsets=[{"id": "omw-en-00031899-r", "start_token": 0, "end_token": 0}],
    )
    result = senses_to_gloss(doc)
    assert result["sentences"][0][0]["gloss"] == ("really" if head_pos == "VERB" else "VERY")
    assert result["indexes"] == [[0, 1]]


@pytest.mark.parametrize("degree", ["Cmp", "Sup"])
def test_degree_intensifier_does_not_change_comparative_or_superlative(degree):
    doc = parsed(
        [("really", "really", "ADV", "advmod", 1), ("better", "good", "ADJ", "ROOT", 1)],
        synsets=[{"id": "omw-en-00031899-r", "start_token": 0, "end_token": 0}],
    )
    doc["tokens"][1]["morph"] = {"Degree": degree}
    result = senses_to_gloss(doc)
    assert [i["gloss"] for i in result["sentences"][0]] == ["really", "better"]


def test_no_longer_idiom_is_not_a_comparative_length():
    doc = parsed(
        [("no", "no", "ADV", "advmod", 1), ("longer", "long", "ADV", "ROOT", 1)],
        synsets=[{"id": "omw-en-00031515-r", "expression": "no longer", "start_token": 0, "end_token": 1}],
    )
    doc["tokens"][1]["morph"] = {"Degree": "Cmp"}
    result = senses_to_gloss(doc)
    assert [i["gloss"] for i in result["sentences"][0]] == ["NOT", "ANYMORE"]
    assert all(i["source"]["tokens"] == doc["tokens"] for i in result["sentences"][0])


@pytest.mark.parametrize("rule", [r for r in rewrites.rules() if len(r["sources"]) == 1], ids=lambda r: r["id"])
def test_same_form_with_different_sense_does_not_rewrite(rule):
    doc = meaning(rule["targets"][0]["form"] if rule["targets"] else "that", "different-selected-sense")
    result = senses_to_gloss(doc)
    assert result["sentences"][0][0]["gloss"] == doc["tokens"][0]["lemma"]
    assert not any(change["rule"].startswith("rewrite:") for change in result["changes"])


@pytest.mark.parametrize("protection", ["missing", "ambiguous", "entity", "ner", "comparative"])
def test_unknown_or_protected_meanings_abstain(protection):
    doc = meaning("unable", "omw-en-00002098-a")
    if protection == "missing":
        doc["synsets"] = []
    elif protection == "ambiguous":
        doc["synsets"].append({**doc["synsets"][0], "id": "other"})
    elif protection == "entity":
        doc["entities"] = [{"id": "Q1", "start_token": 0, "end_token": 0}]
    elif protection == "ner":
        doc["tokens"][0]["ent_type"] = "ORG"
    else:
        doc["tokens"][0]["morph"] = {"Degree": "Cmp"}
    result = senses_to_gloss(doc)
    assert result["indexes"] == [[0]]
    assert result["changes"] == []
    assert result["sentences"][0][0]["gloss"] == "unable"


def test_one_to_many_changes_lookup_senses_and_preserves_source():
    doc = meaning("unable", "omw-en-00002098-a")
    items = senses_to_gloss(doc)["sentences"][0]
    assert [(i["word"], i["gloss"], i["pos"]) for i in items] == [
        ("not", "NOT", "PART"),
        ("can", "CAN", "AUX"),
    ]
    assert [i["rewrite"]["target_index"] for i in items] == [0, 1]
    assert all(i["source"]["tokens"][0]["word"] == "unable" for i in items)
    assert all(i["synsets"][0]["start_token"] == i["synsets"][0]["end_token"] == 0 for i in items)


def test_equivalent_ability_sense_is_replaced_not_appended():
    item = senses_to_gloss(meaning("be able to", "wikidata-en-L1379915-S1"))["sentences"][0][0]
    assert [s["id"] for s in item["synsets"]] == ["wikidata-en-L1888-S1"]
    assert [s["id"] for s in item["source"]["synsets"]] == ["wikidata-en-L1379915-S1"]


def test_negation_and_other_sentence_content_are_preserved():
    doc = parsed(
        [
            ("She", "she", "PRON", "nsubj", 2),
            ("not", "not", "PART", "neg", 2),
            ("unable", "unable", "ADJ", "ROOT", 2),
            (".", ".", "PUNCT", "punct", 2),
        ],
        synsets=[{"id": "omw-en-00002098-a", "start_token": 2, "end_token": 2}],
    )
    result = senses_to_gloss(doc)
    assert [i["gloss"] for i in result["sentences"][0]] == ["she", "not", "NOT", "CAN", "."]
    assert result["indexes"] == [[0, 1, 2, 2, 3]]


def test_semantically_wrong_have_does_not_become_must():
    doc = meaning("have to", "omw-en-02108026-v")
    assert senses_to_gloss(doc)["sentences"][0][0]["gloss"] == "have to"


def test_known_canonical_target_is_a_noop():
    doc = meaning("postpone", "omw-en-02642814-v")
    doc["tokens"][0]["pos"] = "VERB"
    result = senses_to_gloss(doc)
    assert result["changes"] == []
    assert "rewrite" not in result["sentences"][0][0]


def test_single_pass_does_not_chain_targets(monkeypatch):
    first = {
        "id": "first",
        "sources": [{"sense_id": "a"}],
        "targets": [
            {"form": "B", "sense_id": "b", "definition": "B", "pos": "VERB"},
        ],
    }
    second = {
        "id": "second",
        "sources": [{"sense_id": "b"}],
        "targets": [
            {"form": "A", "sense_id": "a", "definition": "A", "pos": "VERB"},
        ],
    }
    monkeypatch.setattr(rewrites, "rules", lambda: [first, second])
    result = senses_to_gloss(meaning("original", "a"))
    assert [i["gloss"] for i in result["sentences"][0]] == ["B"]
    assert [c["rule"] for c in result["changes"]] == ["rewrite:first"]


def obligation_document():
    doc = parsed(
        [
            ("I", "i", "PRON", "nsubj", 1),
            ("have", "have", "VERB", "ROOT", 1),
            ("to", "to", "PART", "aux", 3),
            ("leave", "leave", "VERB", "xcomp", 1),
            (".", ".", "PUNCT", "punct", 1),
        ],
        synsets=[
            {"id": "omw-en-02108026-v", "start_token": 1, "end_token": 1},
            {"id": "wikidata-en-L2985-S1", "start_token": 2, "end_token": 2},
        ],
    )
    doc["tokens"][1]["morph"] = {"Tense": "Pres", "VerbForm": "Fin"}
    doc["tokens"][3]["morph"] = {"VerbForm": "Inf"}
    return doc


def test_experience_have_plus_infinitive_becomes_obligation():
    doc = obligation_document()
    original = deepcopy(doc)
    result = senses_to_gloss(doc)
    assert [i["gloss"] for i in result["sentences"][0]] == ["i", "MUST", "leave", "."]
    assert result["indexes"] == [[0, 1, 3, 4]]
    must = result["sentences"][0][1]
    assert must["synsets"][0]["id"] == "wikidata-en-L1892-S1"
    assert must["rewrite"]["source_candidate_indexes"] == [1, 2]
    assert [s["id"] for s in must["source"]["synsets"]] == [
        "omw-en-02108026-v",
        "wikidata-en-L2985-S1",
    ]
    assert doc == original


@pytest.mark.parametrize(
    "contrast", ["wrong-lemma", "wrong-to-sense", "object", "negative", "past", "perfect", "missing-morphology"]
)
def test_obligation_contrasts_keep_original_meaning(contrast):
    doc = obligation_document()
    if contrast == "wrong-lemma":
        doc["tokens"][1]["lemma"] = "plan"
    elif contrast == "wrong-to-sense":
        doc["synsets"][1]["id"] = "directional-to"
    elif contrast == "past":
        doc["tokens"][1]["morph"]["Tense"] = "Past"
    elif contrast == "perfect":
        doc["tokens"][1]["morph"]["VerbForm"] = "Part"
    elif contrast == "missing-morphology":
        doc["tokens"][1]["morph"] = {}
    else:
        word, lemma, pos, dep = (
            ("not", "not", "PART", "neg") if contrast == "negative" else ("experience", "experience", "NOUN", "dobj")
        )
        doc["tokens"].append({"word": word, "lemma": lemma, "pos": pos, "dep": dep, "head": 1})
        doc["sentences"][0]["end_token"] += 1
    result = senses_to_gloss(doc)
    assert not any(i["gloss"] == "MUST" for s in result["sentences"] for i in s)


def test_negated_complement_does_not_mean_absence_of_obligation():
    doc = obligation_document()
    doc["tokens"].append({"word": "not", "lemma": "not", "pos": "PART", "dep": "neg", "head": 3})
    doc["sentences"][0]["end_token"] += 1
    items = senses_to_gloss(doc)["sentences"][0]
    assert any(i["gloss"] == "MUST" for i in items)
    assert any(i["gloss"] == "not" for i in items)


@pytest.mark.parametrize("subject", ["nobody", "nothing", "none", "neither", "no-one", "no student"])
def test_negative_subject_does_not_lower_obligation(subject):
    doc = obligation_document()
    doc["tokens"][0].update(word=subject, lemma=subject)
    if subject == "no student":
        doc["tokens"][0].update(word="student", lemma="student")
        doc["tokens"].append({"word": "no", "lemma": "no", "pos": "DET", "dep": "det", "head": 0})
        doc["sentences"][0]["end_token"] += 1
    result = senses_to_gloss(doc)
    assert not any(c["rule"] == "rewrite:have-to-obligation" for c in result["changes"])


@pytest.mark.parametrize("negative", [False, True])
def test_controlling_predicate_negation_blocks_obligation(negative):
    doc = parsed(
        [
            ("I", "i", "PRON", "nsubj", 1),
            ("plan", "plan", "VERB", "ROOT", 1),
            ("to", "to", "PART", "aux", 3),
            ("have", "have", "VERB", "xcomp", 1),
            ("to", "to", "PART", "aux", 5),
            ("leave", "leave", "VERB", "xcomp", 3),
        ],
        synsets=[{"id": "wikidata-en-L2985-S1", "start_token": 4, "end_token": 4}],
    )
    for i in (3, 5):
        doc["tokens"][i]["morph"] = {"VerbForm": "Inf"}
    if negative:
        doc["tokens"].append({"word": "not", "lemma": "not", "pos": "PART", "dep": "neg", "head": 1})
        doc["sentences"][0]["end_token"] += 1
    result = senses_to_gloss(doc)
    assert any(c["rule"] == "rewrite:have-to-obligation" for c in result["changes"]) is not negative


@pytest.mark.parametrize("rule", [r for r in rewrites.rules() if r.get("guard") == "predicate-negation"])
@pytest.mark.parametrize("dependency", ["acomp", "amod"])
def test_negative_adjectives_expand_only_as_predicates(rule, dependency):
    doc = parsed(
        [
            ("Seats", "seat", "NOUN", "nsubj", 1),
            ("are", "be", "AUX", "ROOT", 1),
            ("unavailable", "unavailable", "ADJ", dependency, 1 if dependency == "acomp" else 0),
        ],
        synsets=[{"id": rule["sources"][0]["sense_id"], "start_token": 2, "end_token": 2}],
    )
    original = deepcopy(doc)
    result = senses_to_gloss(doc)
    assert any(c["rule"] == "rewrite:" + rule["id"] for c in result["changes"]) is (dependency == "acomp")
    assert doc == original


def test_many_to_many_preserves_all_source_indexes(monkeypatch):
    rule = {
        "id": "two-to-two",
        "sources": [{"sense_id": "a"}, {"sense_id": "b"}],
        "targets": [
            {"form": "C", "sense_id": "c", "definition": "C", "pos": "VERB"},
            {"form": "D", "sense_id": "d", "definition": "D", "pos": "ADV"},
        ],
    }
    monkeypatch.setattr(rewrites, "rules", lambda: [rule])
    doc = parsed(
        [("first", "first", "VERB", "ROOT", 0), ("second", "second", "ADV", "advmod", 0)],
        synsets=[{"id": "a", "start_token": 0, "end_token": 0}, {"id": "b", "start_token": 1, "end_token": 1}],
    )
    result = senses_to_gloss(doc)
    assert [i["gloss"] for i in result["sentences"][0]] == ["C", "D"]
    assert result["indexes"] == [[0, 0]]
    assert all(i["rewrite"]["source_candidate_indexes"] == [0, 1] for i in result["sentences"][0])
    assert result["changes"][0]["source_tokens"] == [0, 1]


def test_explicit_empty_target_list_is_a_traced_omission(monkeypatch):
    monkeypatch.setattr(
        rewrites,
        "rules",
        lambda: [
            {"id": "omit-selected-meaning", "sources": [{"sense_id": "omit-me"}], "targets": []},
        ],
    )
    result = senses_to_gloss(meaning("word", "omit-me"))
    assert result["sentences"] == [[]]
    assert result["indexes"] == [[]]
    assert result["changes"][0]["source_tokens"] == [0]
    assert result["changes"][0]["targets"] == []


@pytest.mark.parametrize("document", [obligation_document(), meaning("unable", "omw-en-00002098-a")])
def test_target_annotations_keep_the_inference_wire_contract(document):
    result = senses_to_gloss(document)
    targets = [item for sentence in result["sentences"] for item in sentence if "rewrite" in item]
    assert targets
    for item in targets:
        for sense in item["synsets"]:
            assert {"id", "definition", "start_token", "end_token", "confidence", "expression"} <= sense.keys()
            assert isinstance(sense["confidence"], float)
            assert 0 <= sense["confidence"] <= 1
            assert sense["expression"] is None


@pytest.mark.parametrize("sense", ["omw-en-02108026-v", "omw-en-02203362-v", "unknown-have-sense", None])
def test_have_sense_is_wildcard_but_infinitive_sense_is_required(sense):
    doc = obligation_document()
    if sense is None:
        doc["synsets"].pop(0)
    else:
        doc["synsets"][0]["id"] = sense
    result = senses_to_gloss(doc)
    assert any(i["gloss"] == "MUST" for s in result["sentences"] for i in s)
    assert result["changes"][-1]["matched_pattern"][0] == {
        "lemma": "have",
        "sense_id": "*",
    }
    doc["synsets"][-1]["id"] = "directional-to"
    assert not any(i["gloss"] == "MUST" for s in senses_to_gloss(doc)["sentences"] for i in s)
