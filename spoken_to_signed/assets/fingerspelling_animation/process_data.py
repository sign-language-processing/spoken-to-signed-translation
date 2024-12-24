import os
from pathlib import Path

import mediapipe as mp
import numpy as np
from matplotlib import image as mpimg
from numpy import ma
from pose_format import PoseHeader, Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeaderDimensions
from pose_format.utils.holistic import holistic_hand_component
from tqdm import tqdm

static_letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k", "l", "m",
                  "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y"]
moving_letters = ["j", "z"]

BASE_URL = "https://fingerspell.net/static/sprites/"

# Manually, to match exactly the shapes in the images
signwriting_mapping = {
    "rest": "S00000",
    "a": "S1f720",
    "b": "S14720",
    "c": "S16d20",
    "d": "S10120",
    "e": "S14a20",
    "f": "S1ce20",
    "g": "S1f000",
    "h": "S11502",
    "i": "S19220",
    "j": "S19220+S2a20c",
    "k": "S14020",
    "l": "S1dc20",
    "m": "S18d20",
    "n": "S11920",
    "o": "S17620",
    "p": "S14051",
    "q": "S1f051",
    "r": "S11a20",
    "s": "S20320",
    "t": "S1fb20",
    "u": "S11520",
    "v": "S10e20",
    "w": "S18620",
    "x": "S10620",
    "y": "S19a20",
    "z": "S10020+S2450a",
}


def download_files():
    file_names = [f"{l}-begin_{l}-end" for l in moving_letters]
    for l1 in static_letters + ["rest"] + [f"{l}-end" for l in moving_letters]:
        for l2 in static_letters + ["rest"] + [f"{l}-begin" for l in moving_letters]:
            file_names.append(f"{l1}_{l2}")

    # download all files to sprites directory
    file_urls = {file_name: f"{BASE_URL}{file_name}.jpg" for file_name in file_names}
    for file_name, file_url in tqdm(file_urls.items()):
        file_path = Path(f"sprites/{file_name}.jpg")
        if not file_path.exists():
            # download using wget, because otherwise we get 406 error
            os.system(f"wget {file_url} -O {file_path}")
        yield file_path


def creat_pose_obj(np_body: np.ndarray):
    dimensions = PoseHeaderDimensions(width=256, height=256, depth=256)
    components = [holistic_hand_component("RIGHT_HAND_LANDMARKS", "XYZC")]
    header = PoseHeader(version=0.2, dimensions=dimensions, components=components)

    pose_body_data = np.expand_dims(np.stack(np_body), axis=1)
    pose_body_conf_shape = pose_body_data.shape[:-1]
    pose_body_conf = np.ones(pose_body_conf_shape)
    body = NumPyPoseBody(data=pose_body_data, confidence=pose_body_conf, fps=60)

    return Pose(header, body)


def pose_hands(file_path: Path):
    pose_path = Path(f"interim_poses/{file_path.stem}.pose")
    if pose_path.exists():
        return

    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0)
    image = mpimg.imread(file_path)
    frames = [image[i:i + 256, :] for i in range(0, image.shape[0], 256)]

    np_arrays = []
    for frame in frames:
        results = hands.process(frame)
        frame_height, frame_width, _ = frame.shape
        landmarks = [results.multi_hand_landmarks[0].landmark[i] for i in range(21)]
        np_landmarks = np.array([[l.x * frame_width, l.y * frame_height, l.z * frame_width] for l in landmarks])
        np_arrays.append(np_landmarks)

    np_body = np.stack(np_arrays)
    pose = creat_pose_obj(np_body)

    with open(pose_path, "wb") as f:
        pose.write(f)


def create_final_poses():
    # 1. read all poses
    poses = {}
    for pose_path in Path("interim_poses").glob("*.pose"):
        with open(pose_path, "rb") as f:
            pose = Pose.read(f.read())
        poses[pose_path.stem] = pose

    # 2. create final poses
    for pose_name, pose in poses.items():
        l_from, l_to = pose_name.split("_")
        if l_from == l_to:  # a static letter
            new_name = signwriting_mapping[l_from]
        elif l_from.endswith("-begin"):  # a dynamic letter
            actual_letter = l_from.removesuffix("-begin")
            new_name = signwriting_mapping[actual_letter]
        elif l_to.endswith("-begin"):  # ends with dynamic letter
            l_to = l_to.removesuffix("-begin")
            extra_pose = poses[f"{l_to}-begin_{l_to}-end"]
            new_pose_data = np.concatenate([pose.body.data, extra_pose.body.data], axis=0)
            new_pose_conf = np.concatenate([pose.body.confidence, extra_pose.body.confidence], axis=0)
            new_pose_body = NumPyPoseBody(data=new_pose_data, confidence=new_pose_conf, fps=pose.body.fps)
            pose = Pose(header=pose.header, body=new_pose_body)

            l_from = l_from.removesuffix("-end")
            new_name = f"{signwriting_mapping[l_from]}-{signwriting_mapping[l_to]}"
        elif l_from.endswith("-end"):  # a dynamic letter to a static letter
            l_from = l_from.removesuffix("-end")
            new_name = f"{signwriting_mapping[l_from]}-{signwriting_mapping[l_to]}"
        else:  # a static letter to a static letter
            new_name = f"{signwriting_mapping[l_from]}-{signwriting_mapping[l_to]}"

        # Copy the pose
        new_pose_data = ma.copy(pose.body.data)
        new_pose_data /= 200 # scale the hand
        new_pose_data -= np.array([1.5, 0.8, 1]) # shift the hand to a good location
        new_pose_body = NumPyPoseBody(data=new_pose_data, confidence=pose.body.confidence, fps=60)
        new_pose = Pose(header=pose.header, body=new_pose_body)

        new_pose_path = Path(f"poses/{new_name}.pose")
        with open(new_pose_path, "wb") as f:
            new_pose.write(f)


if __name__ == "__main__":
    Path("sprites").mkdir(exist_ok=True)
    Path("interim_poses").mkdir(exist_ok=True)
    Path("poses").mkdir(exist_ok=True)

    files = download_files()
    for file in tqdm(files):
        pose_hands(file)

    create_final_poses()
