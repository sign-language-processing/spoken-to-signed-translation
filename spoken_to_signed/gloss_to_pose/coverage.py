import json
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class TokenCoverage:
    word: Optional[str]
    gloss: str
    exact_lexicon_match: bool
    coverage_type: Optional[str] = None  # "lexicon", "language_backup", "fingerspelling_backup", or None
    fingerspelled_keys: Optional[list[str]] = None  # set when coverage_type == "fingerspelling_backup"


@dataclass
class CoverageStats:
    total_tokens: int = 0
    matched_tokens: int = 0
    sentences: list[dict] = field(default_factory=list)

    def add_sentence(self, token_coverages: list[TokenCoverage], text: Optional[str] = None):
        self.sentences.append({"text": text, "tokens": [asdict(tc) for tc in token_coverages]})
        self.total_tokens += len(token_coverages)
        self.matched_tokens += sum(1 for tc in token_coverages if tc.exact_lexicon_match)

    @property
    def fraction(self) -> float:
        return self.matched_tokens / self.total_tokens if self.total_tokens > 0 else 0.0

    def save(self, path: str):
        data = {
            "total_tokens": self.total_tokens,
            "matched_tokens": self.matched_tokens,
            "coverage": self.fraction,
            "sentences": self.sentences,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
