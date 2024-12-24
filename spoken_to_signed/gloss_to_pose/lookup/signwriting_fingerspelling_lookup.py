import string
from functools import lru_cache
from pathlib import Path
from typing import List

import numpy as np
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from signwriting.fingerspelling.fingerspelling import FINGERSPELLING_DIR, get_chars
from signwriting.formats.fsw_to_sign import fsw_to_sign
from spoken_to_signed.gloss_to_pose.concatenate import normalize_pose

from spoken_to_signed.gloss_to_pose import concatenate_poses
from spoken_to_signed.gloss_to_pose.lookup import PoseLookup
from spoken_to_signed.gloss_to_pose.lookup.fingerspelling_lookup import FingerspellingPoseLookup, stretch_pose


def reduce_fsw(fsw: str):
    # Takes fsw stings like M515x532S14020486x469S10020500x502 and only returns the symbols
    symbols = fsw_to_sign(fsw)["symbols"]
    return "+".join(sorted(s["symbol"] for s in symbols))


def get_signwriting_fingerspelling():
    spellings = {}
    for f in FINGERSPELLING_DIR.iterdir():
        if f.is_file():
            # language is 4 pieces: e.g. th-th-tsq-thsl
            _, _, signed_language, _ = f.stem.split('-')
            chars = get_chars(f.stem)
            for char, writings in chars.items():
                chars[char] = [reduce_fsw(writing) for writing in writings]
            chars[""] = ["S00000"]  # null symbol
            spellings[signed_language] = chars
    return spellings


class SignWritingFingerspellingPoseLookup(PoseLookup):
    def __init__(self):
        super().__init__(rows=[])

        self.directory = Path(__file__).parent.parent.parent / "assets" / "fingerspelling_animation"

        # Precompute the alphabets to make the lookup faster
        self.alphabets = get_signwriting_fingerspelling()

        self.naive_spelling = FingerspellingPoseLookup()

    @lru_cache(maxsize=None)
    def get_interpreter(self):
        pose = self.read_pose(f"interpreter.pose")
        return normalize_pose(pose)

    def build_interpreter(self, pose: Pose) -> Pose:
        full_pose = self.get_interpreter()
        num_frames = len(pose.body.data)
        new_data = np.repeat(full_pose.body.data, num_frames, axis=0)
        new_conf = np.repeat(full_pose.body.confidence, num_frames, axis=0)

        for component in pose.header.components:
            component_point_index = pose.header._get_point_index(component.name, component.points[0])

            full_point_index = full_pose.header._get_point_index(component.name, component.points[0])
            num_points = len(component.points)
            new_data[:, :, full_point_index:full_point_index + num_points] = pose.body.data[:, :, component_point_index:component_point_index + num_points]
            new_conf[:, :, full_point_index:full_point_index + num_points] = pose.body.confidence[:, :, component_point_index:component_point_index + num_points]

        # Place elbow in between shoulder and wrist
        right_elbow = full_pose.header._get_point_index("POSE_LANDMARKS", "RIGHT_ELBOW")
        right_shoulder = full_pose.header._get_point_index("POSE_LANDMARKS", "RIGHT_SHOULDER")
        right_wrist = full_pose.header._get_point_index("RIGHT_HAND_LANDMARKS", "WRIST")
        new_data[:, :, right_elbow, 0] = (new_data[:, :, right_shoulder, 0] + new_data[:, :, right_wrist, 0]) / 2

        new_body = NumPyPoseBody(data=new_data, confidence=new_conf, fps=pose.body.fps)
        return Pose(header=full_pose.header, body=new_body)

    def get_transition_names(self, fsw_from, fsw_to):
        if fsw_from == fsw_to:
            yield from fsw_from

        for possible_from in fsw_from:
            for possible_to in fsw_to:
                yield f"{possible_from}-{possible_to}"

    def get_word_poses(self, word: List[str], spoken_language: str, signed_language: str):
        padded_word = [""] + word + [""]
        transitions = [(padded_word[i], padded_word[i + 1]) for i in range(len(padded_word) - 1)]

        pose_so_far = None

        while len(transitions) > 0:
            l_from, l_to = transitions.pop(0)
            print(transitions)

            pose = None
            if l_to in self.alphabets[signed_language]:
                fsw_from = self.alphabets[signed_language][l_from]
                fsw_to = self.alphabets[signed_language][l_to]
                for transition_name in self.get_transition_names(fsw_from, fsw_to):
                    print(transition_name)
                    try:
                        pose = self.read_pose(f"poses/{transition_name}.pose")
                        pose = self.build_interpreter(pose)
                        break
                    except FileNotFoundError:
                        pass

            # Naive concatenation
            if pose is not None:
                if len(transitions) == 0:
                    # hold the last letters longer to make it more readable
                    pose = stretch_pose(pose, 2)

                if pose_so_far is None:
                    pose_so_far = pose
                else:
                    pose_so_far.body.data = np.concatenate([pose_so_far.body.data, pose.body.data], axis=0)
                    pose_so_far.body.confidence = np.concatenate([pose_so_far.body.confidence, pose.body.confidence], axis=0)

            if pose is None:
                print(f"Transition not found")

                # Load naive pose
                pose = self.naive_spelling.get_char_pose(l_to, spoken_language, signed_language)
                # Skip next transition
                if len(transitions) > 0:
                    _, next_l_to = transitions[0]
                    transitions[0] = (next_l_to, next_l_to)

                yield pose_so_far
                pose_so_far = None
                yield pose

        yield pose_so_far

    def lookup(self, word: str, gloss: str, spoken_language: str, signed_language: str, source: str = None) -> Pose:
        if signed_language not in self.alphabets:
            return self.naive_spelling.lookup(word, gloss, spoken_language, signed_language, source)

        word = self.naive_spelling.break_down_word(word.lower(), spoken_language, signed_language)
        poses = list(self.get_word_poses(list(word), spoken_language, signed_language))
        poses = [pose for pose in poses if pose is not None]
        print("num", len(poses))

        return concatenate_poses(poses)


if __name__ == "__main__":
    lookup = SignWritingFingerspellingPoseLookup()
    letters = string.ascii_lowercase
    pose = lookup.lookup(letters, letters, "en", "ase")
    with open("new.pose", "wb") as f:
        pose.write(f)
