import math
from functools import partial

import numpy as np
import scipy.signal
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody


def smooth_non_face(pose: Pose, filter_trajectory) -> Pose:
    # Apply a scipy 1D filter along the time axis to every non-face keypoint, in
    # place; filter_trajectory is invoked as filter_trajectory(array, axis=0). The
    # face is skipped on purpose: smoothing it dampens mouthing and other fast
    # facial expressions that carry meaning, and it has no seam jitter to fix. Its
    # landmarks form one contiguous block, so we filter the keypoints on either
    # side of it in two vectorized calls instead of looping over every point.
    [face_component] = [c for c in pose.header.components if c.name == "FACE_LANDMARKS"]
    face_start = pose.header.get_point_index("FACE_LANDMARKS", face_component.points[0])
    face_end = pose.header.get_point_index("FACE_LANDMARKS", face_component.points[-1])

    data = pose.body.data
    data[:, 0, :face_start] = filter_trajectory(data[:, 0, :face_start], axis=0)
    data[:, 0, face_end:] = filter_trajectory(data[:, 0, face_end:], axis=0)
    return pose


def pose_savgol_filter(pose: Pose) -> Pose:
    return smooth_non_face(pose, partial(scipy.signal.savgol_filter, window_length=3, polyorder=1))


def pose_butterworth_filter(pose: Pose, cutoff: float = 6.0, order: int = 4) -> Pose:
    # Low-pass filter each keypoint trajectory over time to remove the jitter and
    # velocity discontinuities left at the seams between concatenated signs. A
    # zero-phase Butterworth removes high-frequency noise while preserving the
    # sign motion, and smooths transitions better than the light Savitzky-Golay
    # pass (see "Sign Stitching", Walsh et al., BMVC 2024).
    nyquist = pose.body.fps / 2
    wn = min(max(cutoff / nyquist, 1e-3), 0.99)
    b, a = scipy.signal.butter(order, wn, btype="low")

    # filtfilt needs a sequence longer than its edge padding; short clips keep the
    # existing Savitzky-Golay smoothing.
    if pose.body.data.shape[0] <= 3 * max(len(a), len(b)):
        return pose_savgol_filter(pose)

    return smooth_non_face(pose, partial(scipy.signal.filtfilt, b, a))


def create_padding(time: float, example: Pose) -> NumPyPoseBody:
    fps = example.body.fps
    padding_frames = int(time * fps)
    data_shape = example.body.data.shape
    return NumPyPoseBody(
        fps=fps,
        data=np.zeros(shape=(padding_frames, data_shape[1], data_shape[2], data_shape[3])),
        confidence=np.zeros(shape=(padding_frames, data_shape[1], data_shape[2])),
    )


def concatenate_poses(poses: list[Pose], padding: NumPyPoseBody, interpolation="linear") -> Pose:
    # Add padding to all poses except the last one
    for pose in poses[:-1]:
        pose.body.data = np.concatenate((pose.body.data, padding.data))
        pose.body.confidence = np.concatenate((pose.body.confidence, padding.confidence))

    # Concatenate all tensors
    new_data = np.concatenate([pose.body.data for pose in poses])
    new_conf = np.concatenate([pose.body.confidence for pose in poses])
    new_body = NumPyPoseBody(fps=poses[0].body.fps, data=new_data, confidence=new_conf)
    new_body = new_body.interpolate(kind=interpolation)

    # If a point appears in pose1 and pose3 but not pose2, it will be smoothed in pose2, which is ugly
    # TODO: for every conf, if all of it is 0, update it in the new one

    return Pose(header=poses[0].header, body=new_body)


def find_best_connection_point(pose1: Pose, pose2: Pose, window=0.3):
    # window size in seconds, or percentage of the pose, whichever is smaller
    p1_size = math.ceil(min(window * pose1.body.fps, len(pose1.body.data) * window))
    p2_size = math.ceil(min(window * pose2.body.fps, len(pose2.body.data) * window))

    last = np.ma.getdata(pose1.body.data)[len(pose1.body.data) - p1_size :, 0]  # (p1, points, dims)
    first = np.ma.getdata(pose2.body.data)[:p2_size, 0]  # (p2, points, dims)
    last_conf = pose1.body.confidence[len(pose1.body.data) - p1_size :, 0]  # (p1, points)
    first_conf = pose2.body.confidence[:p2_size, 0]  # (p2, points)

    # The seam is about hand/body continuity, but the face is ~130 of the ~180
    # keypoints and barely moves between signs, so it would dominate the distance
    # and drown out the hands. Exclude it from the weighting.
    keep = np.ones(last.shape[1], dtype=bool)
    face = next((c for c in pose1.header.components if c.name == "FACE_LANDMARKS"), None)
    if face is not None:
        face_start = pose1.header.get_point_index("FACE_LANDMARKS", face.points[0])
        keep[face_start : face_start + len(face.points)] = False

    # Confidence-weighted distance between every (last, first) frame pair: a keypoint
    # counts only where it is detected in BOTH frames (weight = product of the two
    # confidences), so undetected keypoints -- whose coordinates are arbitrary -- do
    # not decide the seam. Normalize by the total weight so a pair isn't rewarded for
    # simply having fewer detected keypoints, and skip pairs that share none.
    squared = ((last[:, None] - first[None, :]) ** 2).sum(-1)  # (p1, p2, points)
    weight = last_conf[:, None] * first_conf[None, :] * keep  # (p1, p2, points), face excluded
    total = weight.sum(-1)  # (p1, p2)
    distances = np.where(total > 0, np.sqrt((weight * squared).sum(-1) / np.maximum(total, 1e-8)), np.inf)

    min_index = np.unravel_index(np.argmin(distances), distances.shape)
    last_index = len(pose1.body.data) - p1_size + min_index[0]
    return last_index, min_index[1]


def smooth_concatenate_poses(poses: list[Pose], padding=0.07) -> Pose:
    if len(poses) == 0:
        raise ValueError("No poses to smooth")

    if len(poses) == 1:
        return poses[0]

    start = 0
    for i, pose in enumerate(poses):
        print("Processing", i + 1, "of", len(poses), "...")
        if i != len(poses) - 1:
            end, next_start = find_best_connection_point(poses[i], poses[i + 1])
        else:
            end = len(pose.body.data)
            next_start = None

        pose.body = pose.body[start:end]
        start = next_start

    padding_pose = create_padding(padding, poses[0])
    print("Concatenating...")
    single_pose = concatenate_poses(poses, padding_pose)
    print("Smoothing...")
    return pose_butterworth_filter(single_pose)
