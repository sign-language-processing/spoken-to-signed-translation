import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional


class CoverageType(str, Enum):
    """Coverage type for a gloss token lookup result.

    Inherits from str so that values serialize to JSON naturally and existing
    string comparisons (e.g. == "lexicon") continue to work without changes.
    """

    LEXICON = "lexicon"
    LANGUAGE_BACKUP = "language_backup"
    FINGERSPELLING_BACKUP = "fingerspelling_backup"


@dataclass
class TokenCoverage:
    word: Optional[str]
    gloss: str
    exact_lexicon_match: bool
    coverage_type: Optional[CoverageType] = None
    fingerspelled_keys: Optional[list[str]] = None


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
