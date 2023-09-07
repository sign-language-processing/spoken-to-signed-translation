import functools


@functools.lru_cache(maxsize=None)
def load_spacy_model(model_name: str):
    try:
        import spacy
    except ImportError:
        raise ImportError("Please install spacy. pip install spacy")

    try:
        return spacy.load(model_name)
    except OSError:
        print(f"{model_name} not found. Downloading...")
        import spacy.cli
        spacy.cli.download(model_name)
        return spacy.load(model_name)