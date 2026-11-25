"""Tests for looking up an alternative fingerspelling lexicon.

`FingerspellingPoseLookup` used to hardcode the bundled
`assets/fingerspelling_lexicon` directory, so a caller with its own
monolingual A-Z lexicon had to bypass the constructor with
`__new__` and rebuild `alphabets` by hand. The directory is now an
optional argument, which these tests cover.
"""
import csv
from pathlib import Path

import pytest

from spoken_to_signed.gloss_to_pose.lookup.fingerspelling_lookup import (
    FingerspellingPoseLookup,
)

BUNDLED = Path(__file__).parent.parent / "spoken_to_signed" / "assets" / "fingerspelling_lexicon"


def _copy_one_letter_lexicon(tmp_path: Path) -> Path:
    """
    Build a minimal lexicon with a single letter, reusing a bundled pose.

    We copy a real .pose file rather than synthesising one so the lookup
    exercises the same reader path as production.
    """
    source = BUNDLED / "ase"
    pose_name = next(source.glob("*.pose"))
    target = tmp_path / "lexicon"
    (target / "ase").mkdir(parents=True)
    (target / "ase" / pose_name.name).write_bytes(pose_name.read_bytes())

    with open(target / "index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["path", "spoken_language", "signed_language", "start", "end", "words", "glosses", "priority"]
        )
        writer.writerow([f"ase/{pose_name.name}", "en", "ase", 0, 0, "A", "A", 0])

    return target


def test_defaults_to_bundled_lexicon():
    """Without an argument the bundled lexicon is used, as before."""
    lookup = FingerspellingPoseLookup()
    assert lookup.directory == str(BUNDLED)
    assert len(lookup.alphabets) > 1


def test_accepts_alternative_directory(tmp_path):
    """A caller can point the lookup at its own lexicon directory."""
    directory = _copy_one_letter_lexicon(tmp_path)
    lookup = FingerspellingPoseLookup(str(directory))

    assert lookup.directory == str(directory)
    assert list(lookup.alphabets) == ["en"]
    # make_dictionary_index() lowercases terms, so the alphabet key is "a".
    assert lookup.alphabets["en"]["ase"] == ["a"]


def test_alternative_lexicon_resolves_a_pose(tmp_path):
    """The alternative directory is actually used to load poses."""
    directory = _copy_one_letter_lexicon(tmp_path)
    lookup = FingerspellingPoseLookup(str(directory))

    result = lookup.lookup("A", "A", "en", "ase")

    assert result.pose is not None
    assert len(result.pose.body.data) > 0


def test_bundled_and_alternative_are_independent(tmp_path):
    """Passing a directory must not mutate the bundled default."""
    directory = _copy_one_letter_lexicon(tmp_path)
    FingerspellingPoseLookup(str(directory))

    bundled = FingerspellingPoseLookup()
    assert len(bundled.alphabets) > 1, "bundled lexicon should keep all language pairs"
