"""Tests for smoothing poses that have no face component.

`smooth_non_face` unpacked the FACE_LANDMARKS component with

    [face_component] = [c for c in pose.header.components if c.name == ...]

which raises ValueError on a reduced pose that carries only body and hands
(the common case when faces are stripped to shrink a lexicon). It now
filters every point in one call when no face component is present.
"""
from pathlib import Path

import numpy as np
import pytest
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic

from spoken_to_signed.gloss_to_pose.smoothing import (
    pose_butterworth_filter,
    pose_savgol_filter,
    smooth_non_face,
)

LEXICON = Path(__file__).parent.parent / "assets" / "dummy_lexicon"


def _holistic_pose() -> Pose:
    return Pose.read(next((LEXICON / "sgg").glob("*.pose")).read_bytes())


def _without_face(pose: Pose) -> Pose:
    """Drop the face component, keeping body and both hands."""
    return pose.remove_components(["FACE_LANDMARKS"])


def test_faceless_pose_has_no_face_component():
    """Guard: the fixture really has no face, otherwise the tests prove nothing."""
    pose = _without_face(_holistic_pose())
    assert "FACE_LANDMARKS" not in [c.name for c in pose.header.components]


def test_smooth_non_face_accepts_faceless_pose():
    """The bug: this used to raise ValueError on a face-less pose."""
    pose = _without_face(_holistic_pose())
    result = smooth_non_face(pose, lambda array, axis: array)

    assert result is pose


def test_smooth_non_face_filters_all_points_when_face_missing():
    """
    With no face, every point must be filtered.

    We pass a filter that doubles its input; if any point is skipped its
    values stay unchanged and the assertion fails.
    """
    pose = _without_face(_holistic_pose())
    before = np.array(pose.body.data.filled(np.nan), copy=True)

    smooth_non_face(pose, lambda array, axis: array * 2)

    after = pose.body.data.filled(np.nan)
    np.testing.assert_allclose(after, before * 2)


def test_smoothing_with_face_is_unchanged():
    """A pose that does have a face keeps the original behaviour."""
    pose = _holistic_pose()
    result = smooth_non_face(pose, lambda array, axis: array)
    assert result is pose


@pytest.mark.parametrize("smoother", [pose_savgol_filter, pose_butterworth_filter])
def test_filters_accept_faceless_pose(smoother):
    """The public filter helpers must also survive a face-less pose."""
    pose = _without_face(_holistic_pose())
    smoother(pose)
