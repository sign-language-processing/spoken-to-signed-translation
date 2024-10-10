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
    # read the files (597M)
    with open(file, "rb") as f:
        pose = Pose.read(f.read())

    # trim pose (35% saving)
    pose = trim_pose(pose)

    # log duration
    durations[file.parent.name].append(len(pose.body.data) / pose.body.fps)

    # Run this transformation only once! floating point instability can change the numbers slightly
    if pose.body.data.shape[2] != 576:
        continue

    original_pose = pose

    # reduce holistic (70% saving)
    pose = reduce_holistic(pose)

    # normalize, just for good measure
    pose = normalize_pose(pose)
    scale_normalized_pose(pose)

    # remove appearance to be consistent if the person changes
    pose = remove_appearance(pose)

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

# Heuristically speed up the videos if needed (16MB)
for file in tqdm(Path.cwd().rglob('*.pose')):
    with open(file, "rb") as f:
        pose = Pose.read(f.read())

    interpolate = False

    original_fps = pose.body.fps
    interpolation_fps = pose.body.fps

    if duration_averages[file.parent.name] > 1.1:
        # Practically, changes the speed of the video so that the average is 1~ second
        speed_factor = duration_averages[file.parent.name]
        interpolation_fps /= speed_factor
        interpolate = True

    if original_fps > 30:
        interpolation_fps /= 2
        original_fps /= 2
        interpolate = True

    interpolation_fps = round(interpolation_fps)

    if interpolate:
        # We want to avoid multiple interpolates, so we only interpolate once if needed
        print("Interpolating", file, interpolation_fps)

        # print("Before:", {"fps": pose.body.fps, "duration": len(pose.body.data) / pose.body.fps})
        pose = pose.interpolate(interpolation_fps)
        pose.body.fps = original_fps
        # print("After:", {"fps": pose.body.fps, "duration": len(pose.body.data) / pose.body.fps})
        with open(file, "wb") as f:
            pose.write(f)
