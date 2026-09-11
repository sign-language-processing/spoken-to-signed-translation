from types import SimpleNamespace

import pytest

from spoken_to_signed.text_to_gloss import rules
from spoken_to_signed.text_to_gloss.rules import tokens_to_gloss
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
        ("what color is your car ?", "PRON NOUN AUX PRON NOUN PUNCT", "your car what color ?"),
        ("which book do you want ?", "DET NOUN AUX PRON VERB PUNCT", "you want which book ?"),
        ("how many books do you have ?", "SCONJ ADJ NOUN AUX PRON VERB PUNCT", "you have how many books ?"),
        ("how old are you ?", "SCONJ ADJ AUX PRON PUNCT", "you how old ?"),
        ("what do you think she wants ?", "PRON AUX PRON VERB PRON VERB PUNCT", "what you think she wants ?"),
        ("who Mary is ?", "PRON PROPN AUX PUNCT", "who Mary ?"),
        ("who did the work ?", "PRON VERB DET NOUN PUNCT", "who did work ?"),
        ("who is coming ?", "PRON AUX VERB PUNCT", "who coming ?"),
        ("which book is missing ?", "DET NOUN AUX VERB PUNCT", "which book missing ?"),
        ("what is wrong ?", "PRON AUX ADJ PUNCT", "what wrong ?"),
        ("who is not coming ?", "PRON AUX PART VERB PUNCT", "who not coming ?"),
        ("who is really coming ?", "PRON AUX ADV VERB PUNCT", "who really coming ?"),
        ("which book has been missing ?", "DET NOUN AUX AUX VERB PUNCT", "which book has been missing ?"),
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


def test_empty_input_and_metadata():
    assert tokens_to_gloss([]) == [[]]
    assert tokens_to_gloss([], metadata=[]) == [[]]
    with pytest.raises(ValueError, match="one entry"):
        tokens_to_gloss([], metadata=[{}])


def test_atomic_question_span_moves_without_splitting_or_recreating_it():
    tokens = [GlossItem(w, w) for w in ["how many", "books", "do", "you", "have", "?"]]
    metadata = [{"pos": pos} for pos in ["ADJ", "NOUN", "AUX", "PRON", "VERB", "PUNCT"]]
    [result] = tokens_to_gloss(tokens, metadata=metadata)
    assert all(actual is tokens[i] for actual, i in zip(result, [3, 4, 0, 1, 5]))


@pytest.mark.parametrize("punctuation", [True, False])
def test_text_rules_share_token_rules(monkeypatch, punctuation):
    doc = [
        SimpleNamespace(
            text=word, lemma_=lemma, pos_=pos, is_punct=pos == "PUNCT", morph=SimpleNamespace(to_dict=lambda: {})
        )
        for word, lemma, pos in zip(
            ["What", "is", "your", "name", "?"],
            ["what", "be", "your", "name", "?"],
            ["PRON", "AUX", "PRON", "NOUN", "PUNCT"],
        )
    ]
    monkeypatch.setattr(rules, "load_spacy_model", lambda _: lambda text: doc)
    [result] = rules.text_to_gloss("What is your name?", "en", signed_language="ase", punctuation=punctuation)
    expected = ["your", "name", "What"] + (["?"] if punctuation else [])
    assert [item.word for item in result] == expected


def test_english_rules_do_not_apply_to_other_sign_languages():
    with pytest.raises(ValueError, match="en → ase"):
        rules.text_to_gloss("Hello", "en", signed_language="bfi")


@pytest.mark.parametrize("language", ["de", "fr"])
def test_existing_text_rules_keep_their_path(monkeypatch, language):
    model = object()
    monkeypatch.setattr(rules, "load_spacy_model", lambda _: model)

    def legacy(text, spacy_model, lang, punctuation):
        assert (text, spacy_model, lang, punctuation) == ("input", model, language, True)
        return {"tokens": ["word"], "glosses": ["gloss"]}

    monkeypatch.setattr(rules, "text_to_gloss_given_spacy_model", legacy)
    assert rules.text_to_gloss("input", language, punctuation=True) == [[GlossItem("word", "gloss")]]
