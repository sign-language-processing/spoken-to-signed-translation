#!/usr/bin/env python3
"""Visualize gloss coverage from a JSON file produced by text_to_gloss_to_pose --coverage-stats."""

import argparse
import json

# ANSI color codes
_RESET = "\033[0m"
_COLORS = {
    "lexicon": "\033[92m",           # bright green
    "language_backup": "\033[93m",   # bright yellow
    "fingerspelling_backup": "\033[38;5;214m",  # orange (256-color)
    None: "\033[91m",                # bright red
}

_LEGEND = [
    ("lexicon", "matched via lexicon"),
    ("language_backup", "matched via language backup"),
    ("fingerspelling_backup", "matched via fingerspelling"),
    (None, "not matched"),
]


def _colored(text: str, coverage_type) -> str:
    return f"{_COLORS[coverage_type]}{text}{_RESET}"


def _print_legend():
    print("Legend:")
    for coverage_type, label in _LEGEND:
        print(f"  {_colored('■', coverage_type)} {label}")
    print()


def visualize(coverage_path: str):
    with open(coverage_path, encoding="utf-8") as f:
        data = json.load(f)

    overall_coverage = data.get("coverage", 0.0)
    matched = data.get("matched_tokens", 0)
    total = data.get("total_tokens", 0)

    _print_legend()

    for sentence in data["sentences"]:
        sentence_text = sentence.get("text") or " ".join(t["word"] for t in sentence["tokens"] if t.get("word"))
        colored_glosses = " ".join(
            _colored(t["gloss"], t.get("coverage_type")) for t in sentence["tokens"]
        )
        print(f"Sentence: {sentence_text}")
        print(f"Gloss:    {colored_glosses}")
        print()

    print(f"Overall coverage: {overall_coverage:.3f} ({matched}/{total} tokens matched)")


def main():
    parser = argparse.ArgumentParser(description="Visualize gloss coverage from a JSON coverage file.")
    parser.add_argument("coverage_json", help="Path to the coverage JSON file.")
    args = parser.parse_args()

    visualize(args.coverage_json)


if __name__ == "__main__":
    main()
