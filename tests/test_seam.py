from pathlib import Path

from pose_format import Pose
from pose_format.numpy import NumPyPoseBody

from spoken_to_signed.gloss_to_pose.smoothing import find_best_connection_point

LEXICON = Path(__file__).parent.parent / "assets" / "dummy_lexicon"


def _halves(pose, zero_first_n=0, fill=0.0):
    # Split a pose into two consecutive Pose halves; optionally mark the first N
    # keypoints undetected (confidence 0) in both halves and fill their coordinates.
    n = len(pose.body.data)
    mid = n // 2

    def half(lo, hi):
        data = pose.body.data[lo:hi].copy()
        conf = pose.body.confidence[lo:hi].copy()
        if zero_first_n:
            conf[:, 0, :zero_first_n] = 0
            data[:, 0, :zero_first_n, :] = fill
        return Pose(pose.header, NumPyPoseBody(pose.body.fps, data, conf))

    return half(0, mid), half(mid, n)


def test_returns_valid_indices():
    pose = Pose.read((LEXICON / "sgg" / "essen.pose").read_bytes())
    p1, p2 = _halves(pose)
    last_index, first_index = find_best_connection_point(p1, p2)
    assert 0 <= last_index < len(p1.body.data)
    assert 0 <= first_index < len(p2.body.data)


def test_seam_ignores_zero_confidence_keypoints():
    # Undetected keypoints have arbitrary coordinates; the seam must not depend on
    # them. Zeroing 10 keypoints' confidence and filling them with benign (0) vs
    # garbage (1e6) coordinates must yield the same connection point.
    pose = Pose.read((LEXICON / "sgg" / "essen.pose").read_bytes())
    benign = find_best_connection_point(*_halves(pose, zero_first_n=10, fill=0.0))
    garbage = find_best_connection_point(*_halves(pose, zero_first_n=10, fill=1e6))
    assert benign == garbage
