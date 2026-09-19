"""Prepare atomic gloss candidates from WSD token spans, without dictionary lookup."""


def prepare_tokens(document: dict) -> list[dict]:
    source = document["tokens"]
    annotations = document["synsets"] + document["entities"]
    spans = {(item["start_token"], item["end_token"]) for item in annotations}
    for start, end in spans:
        if type(start) is not int or type(end) is not int or not 0 <= start <= end < len(source):
            raise ValueError(f"Invalid WSD token span: {(start, end)}")
        boundaries = document.get("sentences")
        if boundaries is not None:
            crosses = not any(s["start_token"] <= start <= end <= s["end_token"] for s in boundaries)
        else:
            crosses = start != end and any(token["word"] in {".", "!", "?"} for token in source[start : end + 1])
        if crosses:
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
        heads = [
            i
            for i in range(start, end + 1)
            if source[i].get("dep") == "ROOT" or ("head" in source[i] and not start <= source[i]["head"] <= end)
        ]
        head = heads[0] if len(heads) == 1 else end
        matches = {
            key: [item for item in document[key] if (item["start_token"], item["end_token"]) == (start, end)]
            for key in ("synsets", "entities")
        }
        expression = next((item["expression"] for item in matches["synsets"] if item.get("expression")), None)
        result.append(
            {
                "word": " ".join(part["word"] for part in parts),
                "gloss": expression or " ".join(part["lemma"] for part in parts),
                "pos": source[head]["pos"],
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


def _validate_syntax(document):
    tokens = document["tokens"]
    next_start = 0
    for sentence in document["sentences"]:
        start, end = sentence["start_token"], sentence["end_token"]
        if type(start) is not int or type(end) is not int or start != next_start or not start <= end < len(tokens):
            raise ValueError("sentences must partition the tokens in order (inclusive boundaries)")
        _validate_tree(tokens, start, end)
        next_start = end + 1
    if next_start != len(tokens):
        raise ValueError("sentences must cover every token; use a WSD release exposing syntax")


def _validate_tree(tokens, start, end):
    roots = []
    for index in range(start, end + 1):
        token = tokens[index]
        head = token["head"]
        if type(head) is not int or not start <= head <= end or not token["dep"]:
            raise ValueError("Every token needs a dependency and a head within its sentence")
        if token["dep"] == "ROOT":
            roots.append(index)
            if head != index:
                raise ValueError("A ROOT must point to itself")
    if len(roots) != 1:
        raise ValueError("Each sentence needs exactly one dependency ROOT")
    connected = {roots[0]}
    for index in range(start, end + 1):
        visited = set()
        while index not in connected:
            if index in visited:
                raise ValueError("Dependency heads must form a tree, not a cycle")
            visited.add(index)
            index = tokens[index]["head"]
        connected.update(visited)


def senses_to_gloss(document: dict, *, semantics=None) -> dict:
    """WSD document → aligned ASL candidates. No parsing, lookup of signs, or LLM.

    ``semantics`` optionally implements ``is_time(synset_id)``. Without it, time
    ordering is skipped explicitly. Indexes address prepare_tokens(document),
    while each candidate and change keeps inclusive original-token coordinates.
    """
    from .asl import gloss_sentence

    _validate_syntax(document)
    candidates = prepare_tokens(document)
    positions = {id(item): index for index, item in enumerate(candidates)}
    sentences, changes, notes = [], [], []
    for sentence_index, boundary in enumerate(document["sentences"]):
        items = [c for c in candidates if boundary["start_token"] <= c["start_token"] <= boundary["end_token"]]
        ordered, edits, warnings = gloss_sentence(items, document["tokens"], semantics)
        sentences.append(ordered)
        changes.extend({"sentence": sentence_index, **edit} for edit in edits)
        notes.extend({"sentence": sentence_index, "code": code} for code in warnings)
    return {
        "sentences": sentences,
        "indexes": [[positions[id(c)] for c in s] for s in sentences],
        "changes": changes,
        "notes": notes,
    }
