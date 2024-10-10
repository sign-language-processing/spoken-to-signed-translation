import math
from typing import List

import numpy as np
import scipy.signal
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from scipy.spatial.distance import cdist


def pose_savgol_filter(pose: Pose):
    # If we want this to be faster, here is a possible solution
    # https://stackoverflow.com/questions/75221888/fast-savgol-filter-on-3d-tensor/75406720#75406720

    # Smoothing the face does not result in a good result, so we skip it
    [face_component] = [c for c in pose.header.components if c.name == 'FACE_LANDMARKS']
    face_range = range(
        pose.header._get_point_index('FACE_LANDMARKS', face_component.points[0]),
        pose.header._get_point_index('FACE_LANDMARKS', face_component.points[-1]),
    )

    _, _, points, dims = pose.body.data.shape
    for p in range(points):
        if p not in face_range:
            for d in range(dims):
                pose.body.data[:, 0, p, d] = scipy.signal.savgol_filter(pose.body.data[:, 0, p, d], 3, 1)
    return pose


def create_padding(time: float, example: Pose) -> NumPyPoseBody:
    fps = example.body.fps
    padding_frames = int(time * fps)
    data_shape = example.body.data.shape
    return NumPyPoseBody(fps=fps,
                         data=np.zeros(shape=(padding_frames, data_shape[1], data_shape[2], data_shape[3])),
                         confidence=np.zeros(shape=(padding_frames, data_shape[1], data_shape[2])))


def concatenate_poses(poses: List[Pose], padding: NumPyPoseBody, interpolation='linear') -> Pose:
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

    last_data = pose1.body.data[len(pose1.body.data) - p1_size:]
    first_data = pose2.body.data[:p2_size]

    last_vectors = last_data.reshape(len(last_data), -1)
    first_vectors = first_data.reshape(len(first_data), -1)

    distances_matrix = cdist(last_vectors, first_vectors, 'euclidean')
    min_index = np.unravel_index(np.argmin(distances_matrix, axis=None), distances_matrix.shape)
    last_index = len(pose1.body.data) - p1_size + min_index[0]
    return last_index, min_index[1]


def smooth_concatenate_poses(poses: List[Pose], padding=0.20) -> Pose:
    if len(poses) == 0:
        raise ValueError("No poses to smooth")

    if len(poses) == 1:
        return poses[0]

    start = 0
    for i, pose in enumerate(poses):
        print('Processing', i + 1, 'of', len(poses), '...')
        if i != len(poses) - 1:
            end, next_start = find_best_connection_point(poses[i], poses[i + 1])
        else:
            end = len(pose.body.data)
            next_start = None

        pose.body = pose.body[start:end]
        start = next_start

    padding_pose = create_padding(padding, poses[0])
    print('Concatenating...')
    single_pose = concatenate_poses(poses, padding_pose)
    print('Smoothing...')
    return pose_savgol_filter(single_pose)
