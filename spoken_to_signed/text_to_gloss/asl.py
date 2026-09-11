"""Conservative English→ASL gloss rules over caller-provided tokens.

This is a baseline, not a complete ASL grammar: nonmanuals, spatial agreement,
tense/aspect realization, and clause-level syntax still need richer analysis.
"""

from .types import Gloss


def _omit(item, metadata, following=None, question=False):
    word = (item.word or "").casefold()
    pos = metadata.get("pos")
    if pos == "DET" and word in {"a", "an", "the"}:
        return True
    if pos != "AUX":
        return False
    # Keep past/future/perfect auxiliaries until we can realize their meaning.
    if word in {"am", "is", "are"}:
        return True
    if word in {"do", "does"}:
        # Affirmative do can carry emphasis: "I do like it".
        return question or (following or "").casefold() in {"not", "n't", "n’t"}
    morphology = metadata.get("morphology", [])
    return item.gloss == "be" and any(features.get("Tense") == "Pres" for features in morphology)


def _sentence_to_gloss(sentence, metadata):
    question = (
        len(sentence) > 2
        and sentence[-1].word == "?"
        and (sentence[0].word or "").casefold() in {"what", "who", "where", "when", "why", "how"}
        and metadata[1].get("pos") == "AUX"
        and not any(features.get("pos") in {"CCONJ", "SCONJ"} for features in metadata)
    )
    # Retain original objects, including punctuation, to preserve source alignment.
    gloss = [
        item
        for index, (item, features) in enumerate(zip(sentence, metadata))
        if not _omit(
            item,
            features,
            following=sentence[index + 1].word if index + 1 < len(sentence) else None,
            question=sentence[-1].word == "?"
            and not any(features.get("pos") == "VERB" for features in metadata[:index]),
        )
    ]
    if question:
        gloss.insert(-1, gloss.pop(0))
    return gloss


def tokens_to_gloss(
    tokens: Gloss, language="en", signed_language="ase", *, metadata=None, **unused_kwargs
) -> list[Gloss]:
    """Reorder/drop tokens using aligned ``pos`` and optional ``morphology`` dicts.

    Without metadata, retain tokens unchanged. Multiword items stay atomic. Only
    leading single-word WH questions with an auxiliary and a question mark move;
    relative clauses and WH noun phrases are deliberately left alone.
    """
    if language != "en" or signed_language != "ase":
        raise ValueError("The ASL token rules support only en → ase")
    if metadata is None:
        return [tokens]
    if len(metadata) != len(tokens):
        raise ValueError("metadata must have one entry per token")

    sentences = []
    start = 0
    for end, item in enumerate(tokens, 1):
        if item.word in {".", "!", "?"}:
            sentences.append(_sentence_to_gloss(tokens[start:end], metadata[start:end]))
            start = end
    if start < len(tokens):
        sentences.append(_sentence_to_gloss(tokens[start:], metadata[start:]))
    return sentences
