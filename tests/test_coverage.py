"""Unit tests for CoverageStats and TokenCoverage."""
import pytest

from spoken_to_signed.gloss_to_pose.coverage import CoverageStats, CoverageType, TokenCoverage


def _token(gloss: str, coverage_type=None, exact_lexicon_match: bool = False) -> TokenCoverage:
    return TokenCoverage(word=gloss, gloss=gloss, exact_lexicon_match=exact_lexicon_match, coverage_type=coverage_type)


class TestCoverageStatsFraction:
    def test_no_tokens_returns_zero(self):
        stats = CoverageStats()
        assert stats.fraction == 0.0

    def test_all_matched(self):
        stats = CoverageStats()
        tokens = [_token("a", CoverageType.LEXICON, exact_lexicon_match=True)] * 3
        stats.add_sentence(tokens)
        assert stats.fraction == 1.0

    def test_none_matched(self):
        stats = CoverageStats()
        tokens = [_token("a")] * 4
        stats.add_sentence(tokens)
        assert stats.fraction == 0.0

    def test_partial_match(self):
        stats = CoverageStats()
        tokens = [
            _token("a", CoverageType.LEXICON, exact_lexicon_match=True),
            _token("b", CoverageType.LEXICON, exact_lexicon_match=True),
            _token("c"),
        ]
        stats.add_sentence(tokens)
        assert stats.fraction == pytest.approx(2 / 3)

    def test_accumulates_across_sentences(self):
        stats = CoverageStats()
        stats.add_sentence([_token("a", CoverageType.LEXICON, exact_lexicon_match=True)])
        stats.add_sentence([_token("b"), _token("c")])
        assert stats.total_tokens == 3
        assert stats.matched_tokens == 1
        assert stats.fraction == pytest.approx(1 / 3)


class TestCoverageStatsAddSentence:
    def test_empty_sentence(self):
        stats = CoverageStats()
        stats.add_sentence([])
        assert stats.total_tokens == 0
        assert stats.matched_tokens == 0

    def test_sentence_text_stored(self):
        stats = CoverageStats()
        stats.add_sentence([_token("Danke")], text="Danke.")
        assert stats.sentences[0]["text"] == "Danke."

    def test_sentence_text_none_when_omitted(self):
        stats = CoverageStats()
        stats.add_sentence([_token("Danke")])
        assert stats.sentences[0]["text"] is None


class TestCoverageType:
    def test_values_are_strings(self):
        assert CoverageType.LEXICON == "lexicon"
        assert CoverageType.LANGUAGE_BACKUP == "language_backup"
        assert CoverageType.FINGERSPELLING_BACKUP == "fingerspelling_backup"
        assert CoverageType.UNMATCHED == "unmatched"

    def test_string_comparison(self):
        assert CoverageType.LEXICON == "lexicon"
        assert "lexicon" == CoverageType.LEXICON
