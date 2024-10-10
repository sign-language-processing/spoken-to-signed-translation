# The files are upwards of 500MB. To make them a part of the installable, I would like them to be less
# Therefore, this file reduces holistic (performs pre-proecssing) as well as runs trim_pose ahead of time
from collections import defaultdict
from pathlib import Path

from pose_anonymization.appearance import remove_appearance
from pose_format import Pose
from pose_format.utils.generic import reduce_holistic, get_body_hand_wrist_index, get_hand_wrist_index
from tqdm import tqdm

from spoken_to_signed.gloss_to_pose.concatenate import trim_pose, normalize_pose, scale_normalized_pose

ONLY_RIGHT_HAND = {"ase", "sgg", "gsg"}

durations = defaultdict(list)

for file in tqdm(Path.cwd().rglob('*.pose')):
    # read the files (588MB)
    with open(file, "rb") as f:
        pose = Pose.read(f.read())

    # reduce holistic (185MB)
    pose = reduce_holistic(pose)

    # trim pose (121MB)
    pose = trim_pose(pose)

    # normalize, just for good measure
    pose = normalize_pose(pose)
    scale_normalized_pose(pose)

    # remove appearance to be consistent if the person changes
    pose = remove_appearance(pose)

    # log duration
    durations[file.parent.name].append(len(pose.body.data) / pose.body.fps)

    # Pose estimation is not perfect, so if we don't need the left hand (For selected languages), we can remove it
    if file.parent.name in ONLY_RIGHT_HAND:
        # Remove hand
        left_hand_index = get_hand_wrist_index(pose, "left")
        pose.body.data[:, :, left_hand_index:left_hand_index + 21] = 0
        pose.body.confidence[:, :, left_hand_index:left_hand_index + 21] = 0

        # Remove body wrist
        left_wrist_index = get_body_hand_wrist_index(pose, "left")
        pose.body.data[:, :, left_wrist_index] = 0
        pose.body.confidence[:, :, left_wrist_index] = 0

    # write the file
    with open(file, "wb") as f:
        pose.write(f)

duration_averages = {k: sum(v) / len(v) for k, v in durations.items()}
for language, duration in duration_averages.items():
    print(language, duration)

# Heuristically speed up the videos if needed (47MB)
for file in tqdm(Path.cwd().rglob('*.pose')):
    with open(file, "rb") as f:
        pose = Pose.read(f.read())

    if duration_averages[file.parent.name] > 1:
        speed_factor = duration_averages[file.parent.name]
        pose.body.fps = pose.body.fps * speed_factor
        pose = pose.interpolate(pose.body.fps / speed_factor)

    if pose.body.fps > 30:
        pose = pose.interpolate(pose.body.fps / 2)

    with open(file, "wb") as f:
        pose.write(f)
