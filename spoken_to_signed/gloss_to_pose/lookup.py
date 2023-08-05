import csv
import os
from typing import List

from pose_format import Pose

from spoken_to_signed.text_to_gloss.types import Gloss


class PoseLookup:
    def __init__(self, directory: str):
        self.directory = directory

        with open(os.path.join(directory, 'index.csv'), mode='r', encoding='utf-8') as f:
            self.index = list(csv.DictReader(f))

        self.words_index = self.make_dictionary_index(based_on="words")
        self.glosses_index = self.make_dictionary_index(based_on="glosses")

    def make_dictionary_index(self, based_on: str):
        return {(d[based_on], d['spoken_language'], d['signed_language']): d for d in self.index}

    def read_pose(self, pose_path: str):
        if pose_path.startswith('gs://'):
            import gcsfs

            fs = gcsfs.GCSFileSystem(anon=True)
            with fs.open(pose_path, "rb") as f:
                return Pose.read(f.read())

        if pose_path.startswith('https://'):
            raise NotImplementedError("Can't access pose files from https endpoint")

        pose_path = os.path.join(self.directory, pose_path)
        with open(pose_path, "rb") as f:
            return Pose.read(f.read())

    def lookup(self, word: str, gloss: str, spoken_language: str, signed_language: str) -> Pose:
        lookup_list = [
            (self.words_index, (word, spoken_language, signed_language)),
            (self.glosses_index, (word, spoken_language, signed_language)),
            (self.glosses_index, (gloss, spoken_language, signed_language)),
        ]

        for dict_index, key in lookup_list:
            if key in dict_index:
                return self.read_pose(dict_index[key]["path"])

        raise FileNotFoundError

    def lookup_sequence(self, glosses: Gloss, spoken_language: str, signed_language: str):
        poses: List[Pose] = []
        for word, gloss in glosses:
            try:
                pose = self.lookup(word, gloss, spoken_language, signed_language)
                poses.append(pose)
            except FileNotFoundError:
                pass

        if len(poses) == 0:
            gloss_sequence = ' '.join([f"{word}/{gloss}" for word, gloss in glosses])
            raise Exception(f"No poses found for {gloss_sequence}")

        return poses
