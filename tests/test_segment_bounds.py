from pathlib import Path

from pose_format import Pose

from spoken_to_signed.gloss_to_pose.concatenate import active_signing_span, process_sign
from spoken_to_signed.gloss_to_pose.lookup import PoseLookup

LEXICON = Path(__file__).parent.parent / "assets" / "dummy_lexicon"


def _row(**overrides):
    row = {
        "path": "sgg/essen.pose",
        "spoken_language": "de",
        "signed_language": "sgg",
        "start": "0",
        "end": "0",
        "words": "essen",
        "glosses": "Essen",
        "priority": "0",
    }
    row.update(overrides)
    return row


def _lookup_row(**overrides):
    lookup = PoseLookup(rows=[_row(**overrides)], directory=str(LEXICON))
    return lookup, lookup.words_index["de"]["sgg"]["essen"][0]


def test_no_segment_columns_falls_back():
    # A lexicon without segment_start/segment_end yields no span (elbow fallback).
    lookup, row = _lookup_row()
    _, span = lookup.get_pose(row)
    assert span is None


def test_segment_equal_to_clip_falls_back():
    # Segment bounds equal to the clip bounds also mean "no segmentation".
    lookup, row = _lookup_row(segment_start="0", segment_end="0")
    _, span = lookup.get_pose(row)
    assert span is None


def test_segment_bounds_converted_to_clip_frames():
    # essen.pose is 24 fps (~41.7 ms/frame); the clip starts at 0, so 400 ms -> frame
    # 9 and 2000 ms -> frame 48.
    lookup, row = _lookup_row(segment_start="400", segment_end="2000")
    pose, span = lookup.get_pose(row)
    assert pose.body.fps == 24
    assert span == (9, 48)


def test_process_sign_uses_span():
    # With a span, an interior sign is trimmed exactly to it (no cap, no onset).
    pose = Pose.read((LEXICON / "sgg" / "essen.pose").read_bytes())
    out = process_sign(pose, keep_onset=False, keep_offset=False, max_sign_seconds=None, span=(10, 30))
    assert out.body.data.shape[0] == 20


def test_process_sign_without_span_uses_heuristic():
    pose = Pose.read((LEXICON / "sgg" / "essen.pose").read_bytes())
    first, last = active_signing_span(pose)
    out = process_sign(pose, keep_onset=False, keep_offset=False, max_sign_seconds=None, span=None)
    assert out.body.data.shape[0] == last - first
