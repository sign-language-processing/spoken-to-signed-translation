import functools

from typing import Optional, Tuple


@functools.lru_cache(maxsize=None)
def load_spacy_model(model_names: Tuple[str, ...], disable: Optional[Tuple[str, ...]] = None):
    try:
        import spacy
    except ImportError as e:
        raise ImportError("Please install spacy. pip install spacy") from e

    if disable is None:
        disable = []

    for model_name in model_names:  # Try all models except the last one
        try:
            return spacy.load(model_name, disable=disable)
        except OSError:
            print(f"{model_name} not found")

    # If none of the models worked, download the last one and download if necessary
    last_model = model_names[-1]
    print(f"{last_model} not found. Downloading...")
    import spacy.cli
    spacy.cli.download(last_model)
    return spacy.load(last_model, disable=disable)
