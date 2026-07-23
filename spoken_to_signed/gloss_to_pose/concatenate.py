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
        wrist_index = pose.header._get_point_index("POSE_LANDMARKS", f"{hand}_WRIST")
        elbow_index = pose.header._get_point_index("POSE_LANDMARKS", f"{hand}_ELBOW")
        boundary_start, boundary_end = get_signing_boundary(pose, wrist_index, elbow_index)
        if boundary_start is not None:
            first_frames.append(boundary_start)
        if boundary_end is not None:
            last_frames.append(boundary_end)

    if len(first_frames) == 0:
        return 0, len(pose.body.data)
    return min(first_frames), max(last_frames)


def trim_pose(pose, start=True, end=True):
    if len(pose.body.data) == 0:
        raise ValueError("Cannot trim an empty pose")

    first_frame, last_frame = active_signing_span(pose)
    if not start:
        first_frame = 0
    if not end:
        last_frame = len(pose.body.data)

    pose.body.data = pose.body.data[first_frame:last_frame]
    pose.body.confidence = pose.body.confidence[first_frame:last_frame]
    return pose


def cap_pose_duration(pose: Pose, max_seconds: float) -> Pose:
    # Citation-form dictionary signs are ~12-15x longer than the same sign in
    # fluent signing (mostly preparation, holds and retraction), which is the main
    # reason a naive stitch runs far too long. Speed up any sign longer than
    # max_seconds to that duration, and leave already-short signs untouched.
    fps = pose.body.fps
    num_frames = pose.body.data.shape[0]
    max_frames = round(max_seconds * fps)
    if num_frames <= max_frames or num_frames < 2:
        return pose

    # Resample to max_frames but keep the original fps, so the sign plays back
    # faster (fewer frames at the same rate) rather than at a lower resolution.
    body = pose.body.interpolate(new_fps=fps * max_frames / num_frames, kind="linear")
    body.fps = fps
    return Pose(header=pose.header, body=body)


def slice_pose(pose: Pose, start: int, end: int) -> Pose:
    return Pose(header=pose.header, body=pose.body[start:end])


def join_poses(poses: list[Pose]) -> Pose:
    # Concatenate poses frame-wise, with no transition padding, keeping masks.
    data = np.ma.concatenate([p.body.data for p in poses])
    confidence = np.concatenate([p.body.confidence for p in poses])
    body = NumPyPoseBody(fps=poses[0].body.fps, data=data, confidence=confidence)
    return Pose(header=poses[0].header, body=body)


def concatenate_poses(poses: list[Pose], trim=True, max_sign_seconds: Optional[float] = 0.8) -> Pose:
    if ConcatenationSettings.is_reduce_holistic:
        print("Reducing poses...")
        poses = [reduce_holistic(p) for p in poses]

    print("Normalizing poses...")
    poses = [normalize_pose(p) for p in poses]

    if trim:
        print("Trimming poses...")
        # The sentence's onset (raising the hands into signing space) and offset
        # (lowering them again) are natural rest<->signing transitions we keep at
        # normal speed. Cut those two motions off the first and last signs, trim
        # every sign to its active signing, speed up the over-long ones, then
        # re-attach the onset and offset so only the signs are compressed.
        lead_in_end = active_signing_span(poses[0])[0]
        retraction_start = active_signing_span(poses[-1])[1]
        lead_in = slice_pose(poses[0], 0, lead_in_end) if lead_in_end > 0 else None
        retraction = (
            slice_pose(poses[-1], retraction_start, len(poses[-1].body.data))
            if retraction_start < len(poses[-1].body.data)
            else None
        )

        poses = [trim_pose(p) for p in poses]

        if max_sign_seconds is not None:
            print("Capping sign durations...")
            poses = [cap_pose_duration(p, max_sign_seconds) for p in poses]

        if lead_in is not None:
            poses[0] = join_poses([lead_in, poses[0]])
        if retraction is not None:
            poses[-1] = join_poses([poses[-1], retraction])
    elif max_sign_seconds is not None:
        print("Capping sign durations...")
        poses = [cap_pose_duration(p, max_sign_seconds) for p in poses]

    # Concatenate all poses
    print("Smooth concatenating poses...")
    pose = smooth_concatenate_poses(poses)

    # Correct the wrists (should be after smoothing)
    print("Correcting wrists...")
    pose = correct_wrists(pose)

    # Scale the newly created pose
    print("Scaling pose...")
    normalize_pose_size(pose)

    return pose
