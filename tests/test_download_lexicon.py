from pathlib import Path

from pose_format import Pose

from spoken_to_signed.download_lexicon import LEXICON_INDEX, _segment_loader, _segment_span_ms

LEXICON = Path(__file__).parent.parent / "assets" / "dummy_lexicon"


def test_lexicon_index_has_segment_columns():
    assert "segment_start" in LEXICON_INDEX
    assert "segment_end" in LEXICON_INDEX


def test_segment_loader_none_without_env(monkeypatch):
    # No SEGMENTATION_MODEL_DIR -> segmentation is skipped (no model, no dependency).
    monkeypatch.delenv("SEGMENTATION_MODEL_DIR", raising=False)
    assert _segment_loader() is None


def test_segment_span_none_without_loader():
    pose = Pose.read((LEXICON / "sgg" / "essen.pose").read_bytes())
    assert _segment_span_ms(pose, None) is None
