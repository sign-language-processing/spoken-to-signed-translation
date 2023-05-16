from typing import List, Tuple
from simplemma import text_lemmatizer as simple_lemmatizer, simple_tokenizer
from simplemma.simplemma import LANGLIST


def text_to_gloss(text: str, language: str) -> List[Tuple[str, str]]:
    if language not in LANGLIST:
        raise ValueError(f"Language {language} not supported")

    words = [w.lower() for w in simple_tokenizer(text)]
    lemmas = [w.lower() for w in simple_lemmatizer(text, lang=language)]

    return list(zip(words, lemmas))
