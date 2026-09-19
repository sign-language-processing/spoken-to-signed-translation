"""Rule regression benchmark, NOT a native-ASL translation-quality score.

Run from the repository with spaCy + en_core_web_lg installed:
    PYTHONPATH=. python evaluation/asl/evaluate.py --baseline --split dev
    PYTHONPATH=. python evaluation/asl/evaluate.py --wordnet-url http://localhost:8080

Cases supply selected senses to isolate glossing from WSD errors. Syntax comes
from real spaCy, not hand-adjusted trees. Output is JSON for an auditable ledger.
"""

import argparse
import json
from pathlib import Path

import spacy

from spoken_to_signed.text_to_gloss import rules
from spoken_to_signed.text_to_gloss.types import GlossItem


def document(doc, senses):
    return {
        "tokens": [
            {"word": t.text, "lemma": t.lemma_.lower(), "pos": t.pos_, "morph": t.morph.to_dict(),
             "dep": t.dep_, "head": t.head.i, "ent_type": t.ent_type_}
            for t in doc
        ],
        "sentences": [{"start_token": s.start, "end_token": s.end - 1} for s in doc.sents],
        "synsets": [{"id": senses[t.text], "start_token": t.i, "end_token": t.i}
                    for t in doc if t.text in senses],
        "entities": [],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--split", choices=["dev", "heldout"])
    parser.add_argument("--wordnet-url")
    args = parser.parse_args()
    cases = json.loads(Path(__file__).with_name("cases.json").read_text())
    cases = [c for c in cases if not args.split or c["split"] == args.split]
    nlp = spacy.load("en_core_web_lg")
    if not args.baseline:
        from spoken_to_signed.text_to_gloss.senses import senses_to_gloss
        from spoken_to_signed.text_to_gloss.wordnet import WordNet

        semantics = WordNet(args.wordnet_url) if args.wordnet_url else None
    results = []
    for case, doc in zip(cases, nlp.pipe(c["text"] for c in cases)):
        data = document(doc, case.get("senses", {}))
        if args.baseline:
            sentences = rules.tokens_to_gloss(
                [GlossItem(t.text, t.lemma_) for t in doc],
                metadata=[{"pos": t.pos_, "morphology": [t.morph.to_dict()]} for t in doc],
            )
            output = " | ".join(" ".join(t.word for t in sentence) for sentence in sentences)
        else:
            result = senses_to_gloss(data, semantics=semantics)
            output = " | ".join(" ".join(t["word"] for t in s) for s in result["sentences"])
        results.append({"id": case["id"], "output": output,
                        "pass": output.casefold() in [s.casefold() for s in case["expected"]]})
    print(json.dumps({"passed": sum(r["pass"] for r in results), "total": len(results), "results": results}, indent=2))


if __name__ == "__main__":
    main()
