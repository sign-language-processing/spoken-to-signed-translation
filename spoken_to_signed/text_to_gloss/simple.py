from typing import List

from simplemma import simple_tokenizer
from simplemma import text_lemmatizer as simple_lemmatizer
from simplemma.strategies.dictionaries.dictionary_factory import SUPPORTED_LANGUAGES

from .types import Gloss


def text_to_gloss(text: str, language: str) -> List[Gloss]:
    if language in SUPPORTED_LANGUAGES:
        words = [w.lower() for w in simple_tokenizer(text)]
        lemmas = [w.lower() for w in simple_lemmatizer(text, lang=language)]
    else:
        words = lemmas = [text.lower().split(' ')]

    # TODO add sentence splitting
    return [list(zip(words, lemmas))]
