from typing import List

import numpy as np
from numpy import ma
from pose_format import Pose

from .smoothing import smooth_concatenate_poses


def normalize_pose(pose: Pose) -> Pose:
    return pose.normalize(
        pose.header.normalization_info(p1=("POSE_LANDMARKS", "RIGHT_SHOULDER"), p2=("POSE_LANDMARKS", "LEFT_SHOULDER")))


def reduce_holistic(pose: Pose) -> Pose:
    """
    # import mediapipe as mp
    # points_set = set([p for p_tup in list(mp.solutions.holistic.FACEMESH_CONTOURS) for p in p_tup])
    # face_contours = [str(p) for p in sorted(points_set)]
    # print(face_contours)
    """
    # To avoid installing mediapipe, we just hardcode the face contours given the above code
    face_contours = [
        '0', '7', '10', '13', '14', '17', '21', '33', '37', '39', '40', '46', '52', '53', '54', '55', '58', '61', '63',
        '65', '66', '67', '70', '78', '80', '81', '82', '84', '87', '88', '91', '93', '95', '103', '105', '107', '109',
        '127', '132', '133', '136', '144', '145', '146', '148', '149', '150', '152', '153', '154', '155', '157', '158',
        '159', '160', '161', '162', '163', '172', '173', '176', '178', '181', '185', '191', '234', '246', '249', '251',
        '263', '267', '269', '270', '276', '282', '283', '284', '285', '288', '291', '293', '295', '296', '297', '300',
        '308', '310', '311', '312', '314', '317', '318', '321', '323', '324', '332', '334', '336', '338', '356', '361',
        '362', '365', '373', '374', '375', '377', '378', '379', '380', '381', '382', '384', '385', '386', '387', '388',
        '389', '390', '397', '398', '400', '402', '405', '409', '415', '454', '466'
    ]

    ignore_names = [
        "EAR",
        "NOSE",
        "MOUTH",
        "EYE",  # Face
        "THUMB",
        "PINKY",
        "INDEX",  # Hands
        "KNEE",
        "ANKLE",
        "HEEL",
        "FOOT_INDEX"  # Feet
    ]

    body_component = [c for c in pose.header.components if c.name == 'POSE_LANDMARKS'][0]
    body_no_face_no_hands = [p for p in body_component.points if all(i not in p for i in ignore_names)]

    components = [c.name for c in pose.header.components if c.name != 'POSE_WORLD_LANDMARKS']
    return pose.get_components(components, {"FACE_LANDMARKS": face_contours, "POSE_LANDMARKS": body_no_face_no_hands})


def correct_wrist(pose: Pose, hand: str) -> Pose:
    # print('....')
    wrist_index = pose.header._get_point_index(f'{hand}_HAND_LANDMARKS', 'WRIST')
    wrist = pose.body.data[:, :, wrist_index]
    wrist_conf = pose.body.confidence[:, :, wrist_index]
    # print(hand, "wrist", wrist[-1], wrist[-1].data)
    # print(hand, "wrist_conf", wrist_conf[-1])

    body_wrist_index = pose.header._get_point_index('POSE_LANDMARKS', f'{hand}_WRIST')
    body_wrist = pose.body.data[:, :, body_wrist_index]
    body_wrist_conf = pose.body.confidence[:, :, body_wrist_index]
    # print(hand, "body_wrist", body_wrist[-1])
    # print(hand, "body_wrist_conf", body_wrist_conf[-1])

    new_wrist_data = ma.where(wrist.data == 0, body_wrist, wrist)
    new_wrist_conf = ma.where(wrist_conf == 0, body_wrist_conf, wrist_conf)
    # print(hand, "new_wrist_data", new_wrist_data[-1])
    # print(hand, "new_wrist_conf", new_wrist_conf[-1])

    pose.body.data[:, :, body_wrist_index] = ma.masked_equal(new_wrist_data, 0)
    pose.body.confidence[:, :, body_wrist_index] = new_wrist_conf
    return pose


def correct_wrists(pose: Pose) -> Pose:
    pose = correct_wrist(pose, 'LEFT')
    pose = correct_wrist(pose, 'RIGHT')
    return pose


def trim_pose(pose, start=True, end=True):
    if len(pose.body.data) == 0:
        return pose

    wrist_indexes = [
        pose.header._get_point_index('LEFT_HAND_LANDMARKS', 'WRIST'),
        pose.header._get_point_index('RIGHT_HAND_LANDMARKS', 'WRIST')
    ]
    either_hand = pose.body.confidence[:, 0, wrist_indexes].sum(axis=1) > 0

    first_non_zero_index = np.argmax(either_hand) if start else 0
    last_non_zero_index = (len(either_hand) - np.argmax(either_hand[::-1]) - 1) if end else len(either_hand)

    pose.body.data = pose.body.data[first_non_zero_index:last_non_zero_index]
    pose.body.confidence = pose.body.confidence[first_non_zero_index:last_non_zero_index]
    return pose


def concatenate_poses(poses: List[Pose]) -> Pose:
    print('Reducing poses...')
    poses = [reduce_holistic(p) for p in poses]

    print('Normalizing poses...')
    poses = [normalize_pose(p) for p in poses]

    # Trim the poses to only include the parts where the hands are visible
    print('Trimming poses...')
    poses = [trim_pose(p, i > 0, i < len(poses) - 1) for i, p in enumerate(poses)]

    # Concatenate all poses
    print('Smooth concatenating poses...')
    pose = smooth_concatenate_poses(poses)

    # Correct the wrists (should be after smoothing)
    print('Correcting wrists...')
    pose = correct_wrists(pose)

    # Scale the newly created pose
    print('Scaling pose...')
    new_width = 500
    shift = 1.25
    shift_vec = np.full(shape=(pose.body.data.shape[-1]), fill_value=shift, dtype=np.float32)
    pose.body.data = (pose.body.data + shift_vec) * new_width
    pose.header.dimensions.height = pose.header.dimensions.width = int(new_width * shift * 2)

    return pose
