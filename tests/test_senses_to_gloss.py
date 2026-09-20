from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from spoken_to_signed.server import app
from spoken_to_signed.text_to_gloss.senses import prepare_tokens


def document(words, synsets=(), entities=()):
    return {
        "tokens": [
            {"word": word, "lemma": word.lower(), "pos": "NOUN", "position": i, "morph": {}}
            for i, word in enumerate(words)
        ],
        "synsets": list(synsets),
        "entities": list(entities),
    }


def span(start, end, identifier="omw-en-example", **kwargs):
    return {"id": identifier, "start_token": start, "end_token": end, **kwargs}


def test_repeated_multiword_senses_preserve_ids_morphology_and_source():
    senses = document(
        ["test", "tubes", "and", "test", "tubes"],
        [span(0, 1, expression="test tube"), span(3, 4, expression="test tube")],
    )
    senses["tokens"][1]["morph"] = {"Number": "Plur"}
    original = deepcopy(senses)
    tokens = prepare_tokens(senses)
    assert [token["gloss"] for token in tokens] == ["test tube", "and", "test tube"]
    assert [token["start_token"] for token in tokens] == [0, 2, 3]
    assert tokens[0]["morphology"] == [{}, {"Number": "Plur"}]
    assert tokens[0]["synsets"] == senses["synsets"][:1]
    assert tokens[0]["source"]["tokens"] == senses["tokens"][:2]
    assert tokens[1]["synsets"] == []
    assert senses == original


def test_entity_does_not_inherit_constituent_sense():
    senses = document(["New", "York"], [span(1, 1)], [span(0, 1, 60)])
    [token] = prepare_tokens(senses)
    assert token["word"] == "New York"
    assert token["entities"] == senses["entities"]
    assert token["synsets"] == []
    assert token["source"]["synsets"] == senses["synsets"]


def test_equal_spans_keep_both_entity_and_lexical_senses():
    senses = document(["Ada"], [span(0, 0, "wikidata-en-L1-S2")], [span(0, 0, 7259)])
    [token] = prepare_tokens(senses)
    assert token["synsets"] == senses["synsets"]
    assert token["entities"] == senses["entities"]


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        ((0, 2, 1, 4), ["a", "b c d e"]),
        ((0, 3, 2, 4), ["a b c d", "e"]),
        ((0, 2, 2, 4), ["a b c", "d", "e"]),
    ],
)
def test_overlapping_spans_never_duplicate_or_relabel_words(bounds, expected):
    a, b, c, d = bounds
    tokens = prepare_tokens(document(list("abcde"), [span(a, b)], [span(c, d, 1)]))
    assert [token["word"] for token in tokens] == expected
    assert sum(bool(token["synsets"] or token["entities"]) for token in tokens) == 1


@pytest.mark.parametrize("bounds", [(-1, 0), (0, 8), (1, 0), (True, 1)])
def test_bad_spans_rejected(bounds):
    with pytest.raises(ValueError, match="Invalid WSD"):
        prepare_tokens(document(["a", "b"], [span(*bounds)]))


def test_sentence_crossing_span_rejected():
    with pytest.raises(ValueError, match="sentence boundaries"):
        prepare_tokens(document(["hello", ".", "bye"], [span(0, 2)]))


def test_service_reorders_complete_candidates_and_handles_empty_or_invalid_senses():
    senses = document(["What", "is", "your", "name", "?"], [span(3, 3)])
    for token, pos in zip(senses["tokens"], ["PRON", "AUX", "PRON", "NOUN", "PUNCT"]):
        token["pos"] = pos
    senses["tokens"][1]["lemma"] = "be"
    senses["sentences"] = [{"start_token": 0, "end_token": 4}]
    for token, (dep, head) in zip(
        senses["tokens"], [("attr", 1), ("ROOT", 1), ("poss", 3), ("nsubj", 1), ("punct", 1)]
    ):
        token.update(dep=dep, head=head)
    with TestClient(app) as client:

        def send(value):
            return client.post(
                "/senses-to-gloss",
                json={
                    "senses": value,
                    "spoken_language": "en",
                    "signed_language": "ase",
                },
            )

        response = send(senses)
        assert response.status_code == 200
        result = response.json()
        assert result["indexes"] == [[2, 3, 0]]
        assert [item["word"] for item in result["sentences"][0]] == ["your", "name", "What"]
        assert result["sentences"][0][1]["synsets"] == senses["synsets"]
        assert send({**document([]), "sentences": []}).json() == {
            "sentences": [],
            "indexes": [],
            "changes": [],
            "notes": [],
        }
        assert send({}).status_code == 422
        assert send(document(["a"], [span(0, 2)])).status_code == 422
