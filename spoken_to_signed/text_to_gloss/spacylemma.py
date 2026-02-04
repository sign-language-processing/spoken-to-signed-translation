from .common import load_spacy_model
from .types import Gloss

LANGUAGE_MODELS_SPACY = {
    "de": "de_core_news_lg",
    "fr": "fr_core_news_lg",
    "it": "it_core_news_lg",
    "en": "en_core_web_lg",
}


def text_to_gloss(text: str, language: str, ignore_punctuation: bool = False, **unused_kwargs) -> list[Gloss]:
    if language not in LANGUAGE_MODELS_SPACY:
        raise NotImplementedError(f"Don't know language '{language}'.")

    model_name = LANGUAGE_MODELS_SPACY[language]

    # disable unnecessary components to make lemmatization faster

    spacy_model = load_spacy_model(model_name, disable=("parser", "ner"))

    doc = spacy_model(text)

    glosses = []  # type: Gloss

    for token in doc:
        if ignore_punctuation is True:
            if token.is_punct:
                continue

        gloss = (token.text, token.lemma_)
        glosses.append(gloss)

    return [glosses]
