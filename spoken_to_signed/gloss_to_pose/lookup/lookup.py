import os
from collections import defaultdict
from typing import List

from pose_format import Pose

from spoken_to_signed.text_to_gloss.types import Gloss


class PoseLookup:
    def __init__(self, rows: List, directory: str = None):
        self.directory = directory

        self.words_index = self.make_dictionary_index(rows, based_on="words")
        self.glosses_index = self.make_dictionary_index(rows, based_on="glosses")

        self.file_systems = {}

    def make_dictionary_index(self, rows: List, based_on: str):
        # As an attempt to make the index more compact in memory, we store a dictionary with only what we need
        languages_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for d in rows:
            lower_term = d[based_on].lower()
            languages_dict[d['spoken_language']][d['signed_language']][lower_term].append({
                "path": d['path'],
                "start": d['start'],
                "end": d['end'],
            })
        return languages_dict

    def read_pose(self, pose_path: str):
        if pose_path.startswith('gs://'):
            if 'gcs' not in self.file_systems:
                import gcsfs
                self.file_systems['gcs'] = gcsfs.GCSFileSystem(anon=True)

            with self.file_systems['gcs'].open(pose_path, "rb") as f:
                return Pose.read(f.read())

        if pose_path.startswith('https://'):
            raise NotImplementedError("Can't access pose files from https endpoint")

        if self.directory is None:
            raise ValueError("Can't access pose files without specifying a directory")

        pose_path = os.path.join(self.directory, pose_path)
        with open(pose_path, "rb") as f:
            return Pose.read(f.read())

    def lookup(self, word: str, gloss: str, spoken_language: str, signed_language: str, source: str = None) -> Pose:
        lookup_list = [
            (self.words_index, (spoken_language, signed_language, word)),
            (self.glosses_index, (spoken_language, signed_language, word)),
            (self.glosses_index, (spoken_language, signed_language, gloss)),
        ]

        for dict_index, (spoken_language, signed_language, term) in lookup_list:
            if spoken_language in dict_index:
                if signed_language in dict_index[spoken_language]:
                    lower_term = term.lower()
                    if lower_term in dict_index[spoken_language][signed_language]:
                        rows = dict_index[spoken_language][signed_language][lower_term]
                        # TODO maybe perform additional string match, for correct casing
                        return self.read_pose(rows[0]["path"])

        raise FileNotFoundError

    def lookup_sequence(self, glosses: Gloss, spoken_language: str, signed_language: str, source: str = None):
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
