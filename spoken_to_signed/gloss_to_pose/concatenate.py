import numpy as np
from pose_format import Pose
from pose_format.utils.generic import (
    correct_wrists,
    normalize_pose_size,
    pose_normalization_info,
    reduce_holistic,
)

from spoken_to_signed.gloss_to_pose.smoothing import smooth_concatenate_poses


class ConcatenationSettings:
    is_reduce_holistic = True


def normalize_pose(pose: Pose) -> Pose:
    return pose.normalize(pose_normalization_info(pose.header))


def get_signing_boundary(pose: Pose, wrist_index: int, elbow_index: int) -> tuple[int, int]:
    # Ideally, this could use a sign language detection model.

    pose_length = len(pose.body.data)

    wrist_exists = pose.body.confidence[:, 0, wrist_index] > 0
    first_non_zero_index = np.argmax(wrist_exists).tolist()
    last_non_zero_index = pose_length - np.argmax(wrist_exists[::-1])

    wrist_y = pose.body.data[:, 0, wrist_index, 1]
    elbow_y = pose.body.data[:, 0, elbow_index, 1]

    wrist_above_elbow = wrist_y < elbow_y
    if not np.any(wrist_above_elbow):
        return None, None
    first_active_frame = np.argmax(wrist_above_elbow).tolist()
    last_active_frame = pose_length - np.argmax(wrist_above_elbow[::-1])

    return (max(first_non_zero_index, first_active_frame - 5), min(last_non_zero_index, last_active_frame + 5))


def trim_pose(pose, start=True, end=True):
    if len(pose.body.data) == 0:
        raise ValueError("Cannot trim an empty pose")

    first_frames = []
    last_frames = []

    hands = ["LEFT", "RIGHT"]
    for hand in hands:
        wrist_index = pose.header._get_point_index("POSE_LANDMARKS", f"{hand}_WRIST")
        elbow_index = pose.header._get_point_index("POSE_LANDMARKS", f"{hand}_ELBOW")
        boundary_start, boundary_end = get_signing_boundary(pose, wrist_index, elbow_index)
        if boundary_start is not None:
            first_frames.append(boundary_start)
        if boundary_end is not None:
            last_frames.append(boundary_end)

    if len(first_frames) == 0:
        return pose

    first_frame = min(first_frames)
    last_frame = max(last_frames)

    if not start:
        first_frame = 0
    if not end:
        last_frame = len(pose.body.data)

    pose.body.data = pose.body.data[first_frame:last_frame]
    pose.body.confidence = pose.body.confidence[first_frame:last_frame]
    return pose


def concatenate_poses(poses: list[Pose], trim=True) -> Pose:
    if ConcatenationSettings.is_reduce_holistic:
        print("Reducing poses...")
        poses = [reduce_holistic(p) for p in poses]

    print("Normalizing poses...")
    poses = [normalize_pose(p) for p in poses]

    # Trim the poses to only include the parts where the hands are visible
    if trim:
        print("Trimming poses...")
        poses = [trim_pose(p, i > 0, i < len(poses) - 1) for i, p in enumerate(poses)]

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
