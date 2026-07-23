from typing import NamedTuple, Optional

import numpy as np
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.utils.generic import (
    correct_wrists,
    normalize_pose_size,
    pose_normalization_info,
    reduce_holistic,
)

from spoken_to_signed.gloss_to_pose.smoothing import smooth_concatenate_poses


class SigningBoundary(NamedTuple):
    start: Optional[int]
    end: Optional[int]


class ConcatenationSettings:
    is_reduce_holistic = True


def normalize_pose(pose: Pose) -> Pose:
    return pose.normalize(pose_normalization_info(pose.header))


def get_signing_boundary(pose: Pose, wrist_index: int, elbow_index: int) -> SigningBoundary:
    # Ideally, this could use a sign language detection model.
    pose_length = len(pose.body.data)

    wrist_exists = pose.body.confidence[:, 0, wrist_index] > 0
    first_non_zero_index = np.argmax(wrist_exists).tolist()
    last_non_zero_index = pose_length - np.argmax(wrist_exists[::-1])

    wrist_y = pose.body.data[:, 0, wrist_index, 1]
    elbow_y = pose.body.data[:, 0, elbow_index, 1]

    wrist_above_elbow = wrist_y < elbow_y
    if not np.any(wrist_above_elbow):
        return SigningBoundary(start=None, end=None)
    first_active_frame = np.argmax(wrist_above_elbow).tolist()
    last_active_frame = pose_length - np.argmax(wrist_above_elbow[::-1])

    return SigningBoundary(
        start=max(first_non_zero_index, first_active_frame - 5),
        end=min(last_non_zero_index, last_active_frame + 5),
    )


def active_signing_span(pose: Pose) -> tuple[int, int]:
    # The [first, last) frame range in which either hand is raised and signing.
    # Falls back to the whole clip when no signing is detected.
    first_frames = []
    last_frames = []
    for hand in ("LEFT", "RIGHT"):
        wrist_index = pose.header.get_point_index("POSE_LANDMARKS", f"{hand}_WRIST")
        elbow_index = pose.header.get_point_index("POSE_LANDMARKS", f"{hand}_ELBOW")
        boundary_start, boundary_end = get_signing_boundary(pose, wrist_index, elbow_index)
        if boundary_start is not None:
            first_frames.append(boundary_start)
        if boundary_end is not None:
            last_frames.append(boundary_end)

    if len(first_frames) == 0:
        return 0, len(pose.body)
    return min(first_frames), max(last_frames)


def trim_pose(pose: Pose) -> Pose:
    if len(pose.body) == 0:
        raise ValueError("Cannot trim an empty pose")

    first_frame, last_frame = active_signing_span(pose)
    pose.body.data = pose.body.data[first_frame:last_frame]
    pose.body.confidence = pose.body.confidence[first_frame:last_frame]
    return pose


def cap_pose_duration(pose: Pose, max_seconds: float) -> Pose:
    # Citation-form dictionary signs are ~12-15x longer than the same sign in
    # fluent signing (mostly preparation, holds and retraction), which is the main
    # reason a naive stitch runs far too long. Speed up any sign longer than
    # max_seconds to that duration, and leave already-short signs untouched.
    fps = pose.body.fps
    num_frames = len(pose.body)
    max_frames = round(max_seconds * fps)
    if num_frames <= max_frames or num_frames < 2:
        return pose

    # Resample to max_frames but keep the original fps, so the sign plays back
    # faster (fewer frames at the same rate) rather than at a lower resolution.
    capped = pose.interpolate(fps * max_frames / num_frames, kind="linear")
    capped.body.fps = fps
    return capped


def slice_pose(pose: Pose, start: int, end: int) -> Pose:
    return Pose(header=pose.header, body=pose.body[start:end])


def join_poses(poses: list[Pose]) -> Pose:
    # Concatenate poses frame-wise, with no transition padding, keeping masks.
    if len(poses) == 1:
        return poses[0]
    data = np.ma.concatenate([p.body.data for p in poses])
    confidence = np.concatenate([p.body.confidence for p in poses])
    body = NumPyPoseBody(fps=poses[0].body.fps, data=data, confidence=confidence)
    return Pose(header=poses[0].header, body=body)


def process_sign(
    pose: Pose,
    keep_onset: bool,
    keep_offset: bool,
    max_sign_seconds: Optional[float],
    span: Optional[tuple[int, int]] = None,
) -> Pose:
    # Trim a sign to its active signing span and, if it runs too long, speed it up.
    # The span is a precomputed (segmentation) active-signing range when available,
    # else the elbow heuristic. The sentence's onset (raising the hands into
    # signing space, on the first sign) and offset (lowering them, on the last
    # sign) are natural rest<->signing transitions: keep those at normal speed and
    # re-attach them, so only the sign itself is compressed.
    first, last = span if span is not None else active_signing_span(pose)
    num_frames = len(pose.body)
    onset = slice_pose(pose, 0, first) if keep_onset and first > 0 else None
    offset = slice_pose(pose, last, num_frames) if keep_offset and last < num_frames else None

    sign = slice_pose(pose, first, last)
    if max_sign_seconds is not None:
        sign = cap_pose_duration(sign, max_sign_seconds)

    return join_poses([part for part in (onset, sign, offset) if part is not None])


def _drop_short_spans(flags: np.ndarray, min_len: int) -> np.ndarray:
    # Set any run of True shorter than min_len to False.
    if min_len <= 1:
        return flags
    flags = flags.copy()
    padded = np.concatenate(([False], flags, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])  # alternating run starts/ends
    for start, end in edges.reshape(-1, 2):
        if end - start < min_len:
            flags[start:end] = False
    return flags


def hide_lowered_hands(pose: Pose, threshold: float = 0.15, min_show_seconds: float = 0.2) -> Pose:
    # A hand hanging at rest (wrist low, near the hip) is not signing; drawing its
    # frozen keypoints looks like a detached floating hand. Hide such a hand per
    # frame by zeroing its keypoints' confidence -- PoseVisualizer draws a keypoint
    # only where confidence > 0 -- so the hand follows the real arm while signing
    # and disappears at rest. Height runs from the hip (0) to the shoulder (1);
    # brief appearances (a resting arm that momentarily crept up) are hidden too.
    y = np.ma.getdata(pose.body.data)[:, 0, :, 1]  # vertical position of every keypoint
    min_show_frames = round(min_show_seconds * pose.body.fps)
    for hand in ("LEFT", "RIGHT"):
        component = next((c for c in pose.header.components if c.name == f"{hand}_HAND_LANDMARKS"), None)
        if component is None:
            continue
        wrist = pose.header.get_point_index("POSE_LANDMARKS", f"{hand}_WRIST")
        shoulder = pose.header.get_point_index("POSE_LANDMARKS", f"{hand}_SHOULDER")
        hip = pose.header.get_point_index("POSE_LANDMARKS", f"{hand}_HIP")
        torso = np.median(y[:, hip] - y[:, shoulder])
        if torso == 0:
            continue
        rel_height = (y[:, hip] - y[:, wrist]) / abs(torso)
        shown = _drop_short_spans(rel_height >= threshold, min_show_frames)
        start = pose.header.get_point_index(f"{hand}_HAND_LANDMARKS", component.points[0])
        pose.body.confidence[~shown, 0, start : start + len(component.points)] = 0
    return pose


def concatenate_poses(
    poses: list[Pose],
    trim=True,
    max_sign_seconds: Optional[float] = 0.8,
    hide_idle_hands: bool = True,
    signing_spans: Optional[list[Optional[tuple[int, int]]]] = None,
) -> Pose:
    if ConcatenationSettings.is_reduce_holistic:
        print("Reducing poses...")
        poses = [reduce_holistic(p) for p in poses]

    print("Normalizing poses...")
    poses = [normalize_pose(p) for p in poses]

    if trim:
        print("Trimming poses...")
        last = len(poses) - 1
        spans = signing_spans if signing_spans is not None else [None] * len(poses)
        poses = [
            process_sign(
                pose, keep_onset=i == 0, keep_offset=i == last, max_sign_seconds=max_sign_seconds, span=spans[i]
            )
            for i, pose in enumerate(poses)
        ]
    elif max_sign_seconds is not None:
        print("Capping sign durations...")
        poses = [cap_pose_duration(pose, max_sign_seconds) for pose in poses]

    # Concatenate all poses
    print("Smooth concatenating poses...")
    pose = smooth_concatenate_poses(poses)

    # Correct the wrists (should be after smoothing)
    print("Correcting wrists...")
    pose = correct_wrists(pose)

    # Scale the newly created pose
    print("Scaling pose...")
    normalize_pose_size(pose)

    # Hide hands that hang at rest so they don't render as frozen floating hands
    if hide_idle_hands:
        print("Hiding lowered hands...")
        pose = hide_lowered_hands(pose)

    return pose
