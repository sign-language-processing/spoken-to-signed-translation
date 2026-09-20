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
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from spoken_to_signed.text_to_gloss import rules
from spoken_to_signed.text_to_gloss.types import GlossItem


def document(doc, senses):
    return {
        "tokens": [
            {
                "word": t.text,
                "lemma": t.lemma_.lower(),
                "pos": t.pos_,
                "morph": t.morph.to_dict(),
                "dep": t.dep_,
                "head": t.head.i,
                "ent_type": t.ent_type_,
            }
            for t in doc
        ],
        "sentences": [{"start_token": s.start, "end_token": s.end - 1} for s in doc.sents],
        "synsets": [{"id": senses[t.text], "start_token": t.i, "end_token": t.i} for t in doc if t.text in senses],
        "entities": [],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--split", choices=["dev", "heldout"])
    parser.add_argument("--wordnet-url")
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("cases.json"))
    parser.add_argument("--wsd-url", help="Use actual WSD instead of supplied senses + local spaCy")
    parser.add_argument("--glosser-url", help="Exercise the HTTP container instead of the Python utility")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    cases = [c for c in cases if not args.split or c["split"] == args.split]
    if not args.wsd_url:
        import spacy

        nlp = spacy.load("en_core_web_lg")
        documents = (
            document(doc, case.get("senses", {})) for case, doc in zip(cases, nlp.pipe(c["text"] for c in cases))
        )
    else:

        def disambiguate(case):
            url = args.wsd_url.rstrip("/") + "/disambiguate?" + urlencode({"text": case["text"], "lang": "en"})
            with urlopen(url, timeout=120) as response:
                return json.load(response)

        documents = map(disambiguate, cases)
    if not args.baseline:
        from spoken_to_signed.text_to_gloss.senses import senses_to_gloss
        from spoken_to_signed.text_to_gloss.wordnet import WordNet

        semantics = WordNet(args.wordnet_url) if args.wordnet_url else None
    results = []
    for case, data in zip(cases, documents):
        if args.baseline:
            tokens = [GlossItem(t["word"], t["lemma"]) for t in data["tokens"]]
            sentences = rules.tokens_to_gloss(
                tokens,
                metadata=[{"pos": t["pos"], "morphology": [t["morph"]]} for t in data["tokens"]],
            )
            # Both systems are scored on lexical items, excluding standalone punctuation.
            punctuation = {id(item) for item, t in zip(tokens, data["tokens"]) if t["pos"] == "PUNCT"}
            output = " | ".join(" ".join(t.word for t in sentence if id(t) not in punctuation)
                                for sentence in sentences)
        else:
            if args.glosser_url:
                body = {"spoken_language": "en", "signed_language": "ase", "senses": data}
                request = Request(
                    args.glosser_url.rstrip("/") + "/senses-to-gloss",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urlopen(request, timeout=120) as response:
                    result = json.load(response)
            else:
                result = senses_to_gloss(data, semantics=semantics.is_time if semantics else None)
            output = " | ".join(" ".join(t["word"] for t in s) for s in result["sentences"])
        results.append(
            {
                "id": case["id"],
                "text": case["text"],
                "output": output,
                "pass": output.casefold() in [s.casefold() for s in case["expected"]],
                **({"input": data, "result": result} if args.output and not args.baseline else {}),
            }
        )
    report = {"passed": sum(r["pass"] for r in results), "total": len(results), "results": results}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"passed": report["passed"], "total": report["total"], "output": str(args.output)}))
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
