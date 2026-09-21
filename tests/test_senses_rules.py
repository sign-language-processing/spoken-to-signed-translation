from copy import deepcopy
from unittest.mock import Mock

import pytest

from spoken_to_signed.text_to_gloss.senses import senses_to_gloss


def parsed(rows, synsets=(), entities=(), sentences=None):
    """Explicit syntax isolates rule tests from parser/model changes."""
    return {
        "tokens": [
            {"word": w, "lemma": lemma, "pos": pos, "dep": dep, "head": head} for w, lemma, pos, dep, head in rows
        ],
        "synsets": list(synsets),
        "entities": list(entities),
        "sentences": sentences if sentences is not None else [{"start_token": 0, "end_token": len(rows) - 1}],
    }


def time_document():
    return parsed(
        [
            ("We", "we", "PRON", "nsubj", 1),
            ("meet", "meet", "VERB", "ROOT", 1),
            ("next", "next", "ADJ", "amod", 3),
            ("Tuesday", "tuesday", "PROPN", "npadvmod", 1),
            (".", ".", "PUNCT", "punct", 1),
        ],
        [{"id": "omw-en-15164105-n", "start_token": 3, "end_token": 3}],
    )


def test_punctuation_only_sentence_is_preserved():
    result = senses_to_gloss(parsed([("!", "!", "PUNCT", "ROOT", 0)]))
    assert result["sentences"][0][0]["word"] == "!"
    assert result["sentences"][0][0]["pos"] == "PUNCT"
    assert result["indexes"] == [[0]]
    assert result["changes"] == []


def test_whole_time_phrase_and_provenance():
    doc = time_document()
    original = deepcopy(doc)
    semantics = Mock()
    semantics.matches.return_value = True
    result = senses_to_gloss(doc, semantics=semantics.matches)
    assert result["indexes"] == [[2, 3, 0, 1, 4]]
    assert result["changes"] == [
        {"sentence": 0, "rule": "temporal-frame-first", "source_tokens": [2, 3]},
    ]
    assert doc == original
    semantics.matches.assert_called_once_with("omw-en-15164105-n")


def test_semantics_are_required_not_guessed_from_word():
    assert senses_to_gloss(time_document())["indexes"] == [[0, 1, 2, 3, 4]]
    semantics = Mock()
    semantics.matches.return_value = False
    assert senses_to_gloss(time_document(), semantics=semantics.matches)["indexes"] == [[0, 1, 2, 3, 4]]


def test_atomic_time_phrase_moves_without_splitting():
    doc = time_document()
    doc["synsets"][0]["start_token"] = 2
    result = senses_to_gloss(doc, semantics=lambda _: True)
    assert result["indexes"] == [[2, 0, 1, 3]]
    assert result["sentences"][0][0]["word"] == "next Tuesday"


@pytest.mark.parametrize("protection", ["entity", "ner", "object"])
def test_temporal_looking_entities_and_objects_are_not_frames(protection):
    doc = time_document()
    if protection == "entity":
        doc["entities"] = [{"id": "Q123", "start_token": 2, "end_token": 3}]
    elif protection == "ner":
        doc["tokens"][3]["ent_type"] = "WORK_OF_ART"
    else:
        doc["tokens"][3]["dep"] = "dobj"
    semantics = Mock(matches=lambda _: True)
    assert senses_to_gloss(doc, semantics=semantics.matches)["changes"] == []


@pytest.mark.parametrize("change", ["cycle", "cross-head", "missing-root", "missing-boundary", "overlap-boundary"])
def test_invalid_syntax_is_rejected(change):
    doc = time_document()
    if change == "cycle":
        doc["tokens"][2]["head"] = 3
        doc["tokens"][3]["head"] = 2
    elif change == "cross-head":
        doc["tokens"][0]["head"] = 9
    elif change == "missing-root":
        doc["tokens"][1]["dep"] = "dep"
    elif change == "missing-boundary":
        doc["sentences"] = []
    else:
        doc["sentences"] *= 2
    with pytest.raises(ValueError, match="tree|head|ROOT|sentences"):
        senses_to_gloss(doc)


def test_boundaries_are_authoritative_not_punctuation():
    doc = parsed(
        [
            ("Dr.", "dr.", "PROPN", "compound", 1),
            ("Smith", "smith", "PROPN", "ROOT", 1),
            (".", ".", "PUNCT", "punct", 1),
            ("Go", "go", "VERB", "ROOT", 3),
            ("!", "!", "PUNCT", "punct", 3),
        ],
        sentences=[{"start_token": 0, "end_token": 2}, {"start_token": 3, "end_token": 4}],
    )
    result = senses_to_gloss(doc)
    assert result["indexes"] == [[0, 1, 2], [3, 4]]
    assert [[item["sentence"] for item in sentence] for sentence in result["sentences"]] == [[0, 0, 0], [1, 1]]
    doc["entities"] = [{"id": "Q1", "start_token": 1, "end_token": 3}]
    with pytest.raises(ValueError, match="sentence boundaries"):
        senses_to_gloss(doc)


def test_quoted_punctuation_can_stay_inside_a_single_entity():
    doc = parsed(
        [("Stop", "stop", "VERB", "ROOT", 0), ("!", "!", "PUNCT", "punct", 0)],
        entities=[{"id": "Q1", "start_token": 0, "end_token": 1}],
    )
    assert senses_to_gloss(doc)["indexes"] == [[0]]


def test_unknown_words_are_not_dropped():
    doc = parsed([("Blorf", "blorf", "PROPN", "nsubj", 1), ("dances", "dance", "VERB", "ROOT", 1)])
    assert senses_to_gloss(doc)["indexes"] == [[0, 1]]


def test_entity_subject_can_move_as_a_whole():
    doc = parsed(
        [
            ("Can", "can", "AUX", "aux", 3),
            ("Ada", "ada", "PROPN", "compound", 2),
            ("Lovelace", "lovelace", "PROPN", "nsubj", 3),
            ("help", "help", "VERB", "ROOT", 3),
            ("?", "?", "PUNCT", "punct", 3),
        ],
        entities=[{"id": "Q7259", "start_token": 1, "end_token": 2}],
    )
    result = senses_to_gloss(doc)
    assert result["indexes"] == [[1, 0, 2, 3]]
    assert "question-nonmanuals-not-realized" in result["sentences"][0][0]["notes"]
    assert result["sentences"][0][0]["word"] == "Ada Lovelace"


def test_interrogative_subject_not_misread_as_object():
    doc = parsed(
        [
            ("Which", "which", "DET", "det", 1),
            ("student", "student", "NOUN", "nsubj", 2),
            ("is", "be", "AUX", "ROOT", 2),
            ("the", "the", "DET", "det", 4),
            ("winner", "winner", "NOUN", "attr", 2),
            ("?", "?", "PUNCT", "punct", 2),
        ]
    )
    assert senses_to_gloss(doc)["indexes"] == [[0, 1, 4, 5]]


@pytest.mark.parametrize(
    ("aux", "lemma", "negative", "expected"),
    [
        ("do", "do", "not", [0, 2, 3]),
        ("does", "do", "n't", [0, 2, 3]),
        ("can", "can", "not", [0, 1, 2, 3]),
        ("did", "do", "not", [0, 1, 2, 3]),
    ],
)
def test_negation_keeps_scope_modality_and_past(aux, lemma, negative, expected):
    doc = parsed(
        [
            ("I", "I", "PRON", "nsubj", 3),
            (aux, lemma, "AUX", "aux", 3),
            (negative, "not", "PART", "neg", 3),
            ("go", "go", "VERB", "ROOT", 3),
            (".", ".", "PUNCT", "punct", 3),
        ]
    )
    if aux == "did":
        doc["tokens"][1]["morph"] = {"Tense": "Past"}
    assert senses_to_gloss(doc)["indexes"] == [expected + [4]]


def test_negative_imperative():
    doc = parsed(
        [
            ("Do", "do", "AUX", "aux", 2),
            ("not", "not", "PART", "neg", 2),
            ("go", "go", "VERB", "ROOT", 2),
            ("!", "!", "PUNCT", "punct", 2),
        ]
    )
    assert senses_to_gloss(doc)["indexes"] == [[1, 2, 3]]


@pytest.mark.parametrize("dependency", ["acl", "parataxis", "csubjpass"])
def test_embedded_clause_suppresses_reordering(dependency):
    doc = time_document()
    doc["tokens"][2].update(word="visiting", lemma="visit", pos="VERB", dep=dependency, head=0)
    semantics = Mock(return_value=True)
    result = senses_to_gloss(doc, semantics=semantics)
    assert result["indexes"] == [[0, 1, 2, 3, 4]]
    assert result["notes"] == [{"sentence": 0, "code": "complex-clause-order-preserved"}]
    semantics.assert_not_called()


def test_ambiguous_temporal_senses_abstain():
    doc = time_document()
    doc["synsets"].append({**doc["synsets"][0], "id": "other"})
    semantics = Mock(return_value=True)
    assert senses_to_gloss(doc, semantics=semantics)["indexes"] == [[0, 1, 2, 3, 4]]
    semantics.assert_not_called()
