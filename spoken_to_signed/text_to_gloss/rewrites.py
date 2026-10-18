"""Lower selected WSD meanings to lexical targets after syntactic ordering."""

import json
from functools import cache
from pathlib import Path


@cache
def rules():
    entries = json.loads(Path(__file__).with_suffix(".json").read_text())
    if len(entries) > 100 or len({r["id"] for r in entries}) != len(entries):
        raise ValueError("Expected at most 100 uniquely named transformation rules")
    if any(not r["sources"] or not isinstance(r["targets"], list) for r in entries):
        raise ValueError("Every rule needs selected source senses and an explicit target list")
    if any(
        r.get("guard")
        not in {
            None,
            "affirmative-obligation",
            "object-clause-marker",
            "degree-modifier",
            "present-existential",
            "tolerate-complement",
            "additive-marker",
            "predicate-negation",
        }
        for r in entries
    ):
        raise ValueError("Unknown transformation guard")
    if any(not any(s["sense_id"] != "*" for s in r["sources"]) for r in entries):
        raise ValueError("Every pattern must require at least one specific selected sense")
    if any(s["sense_id"] == "*" and not s.get("lemma") for r in entries for s in r["sources"]):
        raise ValueError("Wildcard senses need a lexical lemma anchor")
    if any(not r.get("guard") and any(s["sense_id"] == "*" for s in r["sources"]) for r in entries):
        raise ValueError("Wildcard senses need a construction guard")
    return sorted(entries, key=lambda r: -len(r["sources"]))


def _negative_obligation_scope(index, tokens):
    """Abstain where a negative subject or controlling predicate obscures modal scope."""
    while True:
        children = [t for t in tokens if t["head"] == index]
        subjects = [i for i, t in enumerate(tokens) if t["head"] == index and t["dep"] == "nsubj"]
        if any(t["dep"] == "neg" for t in children) or any(
            tokens[i]["lemma"].casefold() in {"nobody", "nothing", "none", "neither", "no-one"}
            or any(t["head"] == i and (t["dep"] == "neg" or t["lemma"].casefold() in {"no", "neither"}) for t in tokens)
            for i in subjects
        ):
            return True
        if tokens[index]["dep"] != "xcomp":
            return False
        index = tokens[index]["head"]


def _obligation(items, tokens):
    """Infinitival TO and dependencies establish obligation regardless of HAVE's sense."""
    if len(items) != 2 or any(i["start_token"] != i["end_token"] for i in items):
        return False
    have, marker = (tokens[i["start_token"]] for i in items)
    have_index = items[0]["start_token"]
    verb = tokens[marker["head"]]
    morphology = have.get("morph", {})
    return (
        have["pos"] == "VERB"
        and have["dep"] in {"ROOT", "xcomp", "ccomp", "conj"}
        and (
            morphology.get("VerbForm") == "Inf"
            or morphology.get("VerbForm") == "Fin"
            and morphology.get("Tense") == "Pres"
        )
        and marker["pos"] == "PART"
        and marker["dep"] == "aux"
        and verb["head"] == have_index
        and verb["dep"] == "xcomp"
        and verb["pos"] in {"VERB", "AUX"}
        and verb.get("morph", {}).get("VerbForm") == "Inf"
        and not _negative_obligation_scope(have_index, tokens)
        and not any(
            t["head"] == have_index
            and (
                t["dep"] in {"neg", "dobj", "obj", "iobj", "dative", "auxpass"}
                or t["dep"] == "aux"
                and t["pos"] == "AUX"
            )
            for t in tokens
        )
    )


def _matches(rule, items, tokens):
    if len(items) != len(rule["sources"]):
        return False
    if any(left["end_token"] + 1 != right["start_token"] for left, right in zip(items, items[1:])):
        return False  # Do not consume across an omission, reordered phrase or punctuation.
    for item, source in zip(items, rule["sources"]):
        if item["entities"] or any(t.get("ent_type") for t in item["source"]["tokens"]):
            return False
        # An atomic idiom owns its internal grammar: NO LONGER is temporal,
        # while a singleton LONGER still carries comparative degree.
        if item["start_token"] == item["end_token"] and any(
            m.get("Degree") in {"Cmp", "Sup"} for m in item["morphology"]
        ):
            return False
        if source.get("lemma") and (
            item["start_token"] != item["end_token"]
            or tokens[item["start_token"]]["lemma"].casefold() != source["lemma"].casefold()
        ):
            return False
        if source["sense_id"] != "*" and [s["id"] for s in item["synsets"]] != [source["sense_id"]]:
            return False
    return _guard_matches(rule, items, tokens)


def _guard_matches(rule, items, tokens):
    if rule.get("guard") == "predicate-negation":
        return (
            len(items) == 1
            and items[0]["start_token"] == items[0]["end_token"]
            and tokens[items[0]["start_token"]]["pos"] == "ADJ"
            and tokens[items[0]["start_token"]]["dep"] in {"ROOT", "acomp", "attr", "xcomp"}
        )
    if rule.get("guard") == "tolerate-complement":
        return _tolerate_complement(items, tokens)
    if rule.get("guard") == "additive-marker":
        following = items[0]["end_token"] + 1
        return following == len(tokens) or tokens[following]["lemma"].casefold() != "as"
    if rule.get("guard") == "present-existential":
        return _existential(items, tokens)
    if rule.get("guard") in {"object-clause-marker", "degree-modifier"}:
        if len(items) != 1 or items[0]["start_token"] != items[0]["end_token"]:
            return False
        marker = tokens[items[0]["start_token"]]
        head = tokens[marker["head"]]
        if rule["guard"] == "degree-modifier":
            return (
                marker["pos"] == "ADV"
                and marker["dep"] == "advmod"
                and head["pos"] in {"ADJ", "ADV"}
                and head.get("morph", {}).get("Degree") not in {"Cmp", "Sup"}
            )
        return (
            marker["pos"] == "SCONJ"
            and marker["dep"] == "mark"
            and head["pos"] in {"VERB", "AUX"}
            and head["dep"] == "ccomp"
        )
    return rule.get("guard") is None or _obligation(items, tokens)


def _tolerate_complement(items, tokens):
    if len(items) != 2:
        return False
    phrase, marker = items
    parts = tokens[phrase["start_token"] : phrase["end_token"] + 1]
    if [t["lemma"].casefold() for t in parts] != ["put", "up"]:
        return False
    verb = phrase["start_token"]
    preposition = tokens[marker["start_token"]]
    return (
        parts[0]["pos"] == "VERB"
        and parts[1]["dep"] == "prt"
        and parts[1]["head"] == verb
        and preposition["pos"] == "ADP"
        and preposition["dep"] == "prep"
        and preposition["head"] == verb
        and any(t["head"] == marker["start_token"] and t["dep"] == "pobj" for t in tokens)
    )


def _existential(items, tokens):
    if len(items) != 2 or any(i["start_token"] != i["end_token"] for i in items):
        return False
    there, be = (tokens[i["start_token"]] for i in items)
    be_index = items[1]["start_token"]
    children = [t for t in tokens if t["head"] == be_index]
    return (
        there["pos"] == "PRON"
        and there["dep"] == "expl"
        and there["head"] == be_index
        and be["pos"] in {"VERB", "AUX"}
        and be["dep"] == "ROOT"
        and be.get("morph", {}).get("Tense") == "Pres"
        and be.get("morph", {}).get("VerbForm") == "Fin"
        and any(t["dep"] in {"attr", "nsubj"} and t["pos"] in {"NOUN", "PROPN"} for t in children)
        and not any(t["dep"] in {"neg", "aux", "auxpass", "ccomp", "xcomp"} for t in children)
        and not any(t["word"] == "?" for t in children)
        and not any(
            t["lemma"] in {"no", "none", "nothing", "nobody", "no-one"}
            and (t["head"] == be_index or tokens[t["head"]]["head"] == be_index)
            for t in tokens
        )
    )


def _emit(rule, items, positions, document):
    start, end = items[0]["start_token"], items[-1]["end_token"]
    origins = [positions[id(item)] for item in items]
    source = {
        "tokens": document["tokens"][start : end + 1],
        **{
            key: [a for a in document[key] if start <= a["start_token"] <= a["end_token"] <= end]
            for key in ("synsets", "entities")
        },
    }
    confidence = min((s.get("confidence", 1.0) for item in items for s in item["synsets"]), default=1.0)
    output = [
        {
            **items[0],
            "word": target["form"].lower(),
            "gloss": target["form"],
            "pos": target["pos"],
            "start_token": start,
            "end_token": end,
            "morphology": [t.get("morph", {}) for t in source["tokens"]],
            "source": source,
            "synsets": [
                {
                    "id": target["sense_id"],
                    "definition": target["definition"],
                    "start_token": start,
                    "end_token": end,
                    "confidence": confidence,
                    "expression": None,
                }
            ],
            "rewrite": {"rule": rule["id"], "source_candidate_indexes": origins, "target_index": index},
        }
        for index, target in enumerate(rule["targets"])
    ]
    change = {
        "rule": "rewrite:" + rule["id"],
        "source_tokens": list(range(start, end + 1)),
        "source_candidate_indexes": origins,
        "source_sense_ids": [[s["id"] for s in item["synsets"]] for item in items],
        "matched_pattern": rule["sources"],
        "targets": [{"form": t["form"], "sense_id": t["sense_id"]} for t in rule["targets"]],
    }
    return output, change


def rewrite(items, positions, document):
    """One longest-pattern-first M:N pass; generated targets cannot trigger another rule.

    Unknown/ambiguous senses, named entities and degree-marked singleton predicates abstain.
    Target senses replace lookup annotations; original annotations stay in source.
    """
    output, changes, index = [], [], 0
    while index < len(items):
        rule = next(
            (
                r
                for r in rules()
                if _matches(
                    r,
                    items[index : index + len(r["sources"])],
                    document["tokens"],
                )
            ),
            None,
        )
        if rule is None:
            output.append(items[index])
            index += 1
            continue
        matched = items[index : index + len(rule["sources"])]
        targets = rule["targets"]
        noop = len(matched) == len(targets) == 1 and (
            matched[0]["gloss"].casefold() == targets[0]["form"].casefold()
            and matched[0]["synsets"][0]["id"] == targets[0]["sense_id"]
            and matched[0]["pos"] == targets[0]["pos"]
        )
        if noop:
            output.extend(matched)
        else:
            replacement, change = _emit(rule, matched, positions, document)
            output.extend(replacement)
            changes.append(change)
        index += len(matched)
    return output, changes
