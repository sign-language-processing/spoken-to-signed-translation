"""Conservative English→ASL ordering rules over already parsed WSD candidates.

This produces a lexical plan, not complete ASL: spatial agreement, classifiers,
aspect and nonmanuals still need a realization stage. See evaluation/asl/README.md.
"""

from .rules import _asl_question_length, _omit_asl
from .types import GlossItem


def _head(item, tokens):
    start, end = item["start_token"], item["end_token"]
    heads = [i for i in range(start, end + 1) if tokens[i]["head"] == i or not start <= tokens[i]["head"] <= end]
    return heads[0] if len(heads) == 1 else None


def _subtree(root, tokens, members):
    found = {root}
    while True:
        children = {i for i in members if tokens[i]["head"] in found}
        if children <= found:
            return found
        found |= children


def _protected(item):
    return bool(item["entities"]) or any(t.get("ent_type") not in {None, "", "DATE", "TIME"}
                                         for t in item["source"]["tokens"])


def _drop(item, tokens, root, question):
    # A multiword meaning or named entity is indivisible, even if it contains "the".
    if item["start_token"] != item["end_token"] or _protected(item):
        return False
    index = item["start_token"]
    token = tokens[index]
    if token["lemma"] == "be":
        # Preserve existential, passive, progressive and elliptical constructions.
        if token["dep"] != "ROOT" or any(t["dep"] == "expl" and t["head"] == index for t in tokens):
            return False
        if not any(t["head"] == index and t["dep"] in {"attr", "acomp", "prep", "advmod"} for t in tokens):
            return False
    negatives = [t["word"] for t in tokens if t["head"] == token["head"] and t["dep"] == "neg"]
    return _omit_asl(GlossItem(item["word"], token["lemma"]), item,
                     following=negatives[0] if negatives else None,
                     question=question and token["head"] == root and token["dep"] == "aux")


def gloss_sentence(items, tokens, semantics=None):
    """Ordered rules: safe omission → temporal frame → simple question ordering.

    Uncertain or multi-clause syntax keeps source order. We never duplicate items,
    split a WSD span, invent a sense, or drop a negation/modal/content word.
    """
    members = set(range(items[0]["start_token"], items[-1]["end_token"] + 1))
    root = next(i for i in members if tokens[i]["dep"] == "ROOT")
    question = any(tokens[i]["word"] == "?" for i in members)
    complex_clause = any(tokens[i]["dep"] in {"ccomp", "xcomp", "advcl", "relcl", "csubj"}
                         or (tokens[i]["dep"] == "conj" and tokens[i]["pos"] in {"VERB", "AUX"}) for i in members)
    notes = ["complex-clause-order-preserved"] if complex_clause else []
    if semantics is None:
        notes.append("temporal-semantics-unavailable")
    if question:
        notes.append("question-nonmanuals-not-realized")
    edits = []
    order = []
    for item in items:
        if _drop(item, tokens, root, question):
            edits.append({"rule": "omit-function-word", "source_tokens": [item["start_token"]]})
        else:
            order.append(item)
    if not complex_clause:
        length = _asl_question_length([GlossItem(i["word"], i["gloss"]) for i in items], items)
        prefix_ids = {id(i) for i in items[:length]}
        prefix = [i for i in order if id(i) in prefix_ids]
        if prefix:
            order = [i for i in order[:-1] if id(i) not in prefix_ids] + prefix + order[-1:]
            edits.append({"rule": "wh-final", "source_tokens": sorted(
                p for i in prefix for p in range(i["start_token"], i["end_token"] + 1))})
    return order, edits, notes
