"""Prepare atomic gloss candidates from WSD token spans, without dictionary lookup."""


def prepare_tokens(document: dict) -> list[dict]:
    source = document["tokens"]
    annotations = document["synsets"] + document["entities"]
    spans = {(item["start_token"], item["end_token"]) for item in annotations}
    for start, end in spans:
        if type(start) is not int or type(end) is not int or not 0 <= start <= end < len(source):
            raise ValueError(f"Invalid WSD token span: {(start, end)}")
        if start != end and any(token["word"] in {".", "!", "?"} for token in source[start : end + 1]):
            raise ValueError("A WSD span must not cross sentence boundaries")

    # Keep widest meanings atomic; ties prefer the earlier span. Never duplicate words.
    selected, covered = {}, set()
    for start, end in sorted(spans, key=lambda span: (span[0] - span[1], span[0])):
        positions = set(range(start, end + 1))
        if not covered.intersection(positions):
            selected[start] = end
            covered.update(positions)

    result, start = [], 0
    while start < len(source):
        end = selected.get(start, start)
        parts = source[start : end + 1]
        matches = {
            key: [item for item in document[key] if (item["start_token"], item["end_token"]) == (start, end)]
            for key in ("synsets", "entities")
        }
        expression = next((item["expression"] for item in matches["synsets"] if item.get("expression")), None)
        result.append(
            {
                "word": " ".join(part["word"] for part in parts),
                "gloss": expression or " ".join(part["lemma"] for part in parts),
                "pos": parts[-1]["pos"],
                "morphology": [part.get("morph", {}) for part in parts],
                "start_token": start,
                "end_token": end,
                # Retain source tokens and contained annotations for future constituent fallback.
                # Only exact-boundary annotations above describe the whole translation unit.
                "source": {
                    "tokens": parts,
                    **{
                        key: [
                            item for item in document[key] if start <= item["start_token"] <= item["end_token"] <= end
                        ]
                        for key in ("synsets", "entities")
                    },
                },
                **matches,
            }
        )
        start = end + 1
    return result
