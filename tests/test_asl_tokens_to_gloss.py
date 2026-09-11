import pytest

from spoken_to_signed.text_to_gloss.asl import tokens_to_gloss
from spoken_to_signed.text_to_gloss.types import GlossItem


def gloss(words, positions, lemmas=None, morphology=None):
    tokens = [GlossItem(word, lemma) for word, lemma in zip(words.split(), (lemmas or words).split())]
    metadata = [{"pos": pos} for pos in positions.split()]
    if morphology:
        for index, features in morphology.items():
            metadata[index]["morphology"] = [features]
    result = tokens_to_gloss(tokens, metadata=metadata)
    assert all(any(item is original for original in tokens) for sentence in result for item in sentence)
    assert len({id(item) for sentence in result for item in sentence}) == sum(map(len, result))
    return [[item.word for item in sentence] for sentence in result]


@pytest.mark.parametrize(
    ("words", "positions", "expected"),
    [
        ("what is your name ?", "PRON AUX PRON NOUN PUNCT", "your name what ?"),
        ("my name is amit", "PRON NOUN AUX PROPN", "my name amit"),
        ("where do you live ?", "ADV AUX PRON VERB PUNCT", "you live where ?"),
        ("I do not like the book .", "PRON AUX PART VERB DET NOUN PUNCT", "I not like book ."),
        ("I do the work .", "PRON VERB DET NOUN PUNCT", "I do work ."),
        ("I do like it .", "PRON AUX VERB PRON PUNCT", "I do like it ."),
        ("do you know I do like it ?", "AUX PRON VERB PRON AUX VERB PRON PUNCT", "you know I do like it ?"),
        (
            "what is your name , and where do you live ?",
            "PRON AUX PRON NOUN PUNCT CCONJ ADV AUX PRON VERB PUNCT",
            "what your name , and where you live ?",
        ),
        ("she has a book .", "PRON VERB DET NOUN PUNCT", "she has book ."),
        ("she has read the book .", "PRON AUX VERB DET NOUN PUNCT", "she has read book ."),
        ("they were here .", "PRON AUX ADV PUNCT", "they were here ."),
        ("she will be here .", "PRON AUX AUX ADV PUNCT", "she will be here ."),
        ("go where we live .", "VERB SCONJ PRON VERB PUNCT", "go where we live ."),
        ("what color is your car ?", "DET NOUN AUX PRON NOUN PUNCT", "what color your car ?"),
        ("who did the work ?", "PRON VERB DET NOUN PUNCT", "who did work ?"),
        ("what is your name", "PRON AUX PRON NOUN", "what your name"),
    ],
)
def test_conservative_rules(words, positions, expected):
    assert gloss(words, positions) == [expected.split()]


def test_each_question_stays_in_its_sentence():
    assert gloss("where are you ? my name is amit .", "ADV AUX PRON PUNCT PRON NOUN AUX PROPN PUNCT") == [
        ["you", "where", "?"],
        ["my", "name", "amit", "."],
    ]


def test_contract_and_atomic_spans_without_metadata():
    tokens = [GlossItem("the United States", "United-States"), GlossItem("example.com", "example.com")]
    assert tokens_to_gloss(tokens) == [tokens]
    assert tokens_to_gloss(tokens, metadata=[{"pos": "PROPN"}, {"pos": "NOUN"}])[0][0] is tokens[0]
    with pytest.raises(ValueError, match="one entry"):
        tokens_to_gloss(tokens, metadata=[])
    with pytest.raises(ValueError, match="only en"):
        tokens_to_gloss(tokens, language="de")


def test_contracted_copula_uses_lemma_and_morphology():
    assert gloss("she 's here", "PRON AUX ADV", "she be here", {1: {"Tense": "Pres"}}) == [["she", "here"]]
    assert gloss("she 's eaten", "PRON AUX VERB", "she have eat", {1: {"Tense": "Pres"}}) == [["she", "'s", "eaten"]]
