import functools

from typing import Optional, Tuple


@functools.lru_cache(maxsize=None)
def load_spacy_model(model_name: str, disable: Optional[Tuple[str, ...]] = None):
    try:
        import spacy
    except ImportError:
        raise ImportError("Please install spacy. pip install spacy")

    if disable is None:
        disable = []

    try:
        return spacy.load(model_name, disable=disable)
    except OSError:
        print(f"{model_name} not found. Downloading...")
        import spacy.cli
        spacy.cli.download(model_name)
        return spacy.load(model_name, disable=disable)
