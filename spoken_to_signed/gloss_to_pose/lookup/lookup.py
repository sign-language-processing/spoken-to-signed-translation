import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List

from pose_format import Pose

from spoken_to_signed.gloss_to_pose.languages import LANGUAGE_BACKUP
from spoken_to_signed.gloss_to_pose.lookup.lru_cache import LRUCache
from spoken_to_signed.text_to_gloss.types import Gloss

from spoken_to_signed.gloss_to_pose.lookup.gloss_normalization_helpers import (
    get_progressive_gloss_normalizers,
    should_normalize_integer_token
)

class PoseLookup:
    def __init__(self, rows: List,
                 directory: str = None,
                 backup: "PoseLookup" = None,
                 cache: LRUCache = None):
        self.directory = directory

        self.words_index = self.make_dictionary_index(rows, based_on="words")
        self.glosses_index = self.make_dictionary_index(rows, based_on="glosses")

        self.backup = backup

        self.file_systems = {}
        self.cache = cache if cache is not None else LRUCache()

    def make_dictionary_index(self, rows: List, based_on: str):
        # As an attempt to make the index more compact in memory, we store a dictionary with only what we need
        languages_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for d in rows:
            term = d[based_on]
            lower_term = term.lower()
            languages_dict[d['spoken_language']][d['signed_language']][lower_term].append({
                "path": d['path'],
                "term": term,
                "start": int(d['start']),
                "end": int(d['end']),
                "priority": int(d['priority']),
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

    def get_pose(self, row):
        # Manage pose cache
        cached_pose = self.cache.get(row["path"])
        if cached_pose is None:
            pose = self.read_pose(row["path"])
            self.cache.set(row["path"], pose)
        pose = self.cache.get(row["path"])

        frame_time = 1000 / pose.body.fps
        start_frame = math.floor(row["start"] // frame_time)
        end_frame = math.ceil(row["end"] // frame_time) if row["end"] > 0 else -1
        return Pose(pose.header, pose.body[start_frame:end_frame])

    def get_best_row(self, rows, term: str):
        # Sort by priority: lower is "better"
        rows = sorted(rows, key=lambda x: x["priority"])
        # String match exact term
        for row in rows:
            if term == row["term"]:
                return row
        # Return the highest priority row
        return rows[0]

    def lookup(self, word: str, gloss: str, spoken_language: str, signed_language: str, source: str = None) -> Pose:
        
        # List of progressive transformations applied ONLY in the main language
        preprocess_steps = get_progressive_gloss_normalizers()

        current_gloss = gloss

        # Attempts within the main language
        for step_idx, step_fn in enumerate(preprocess_steps):
            # Apply cumulative transformations except for attempt #0
            if step_fn is not None:
                current_gloss = step_fn(current_gloss)

            lookup_list = [
                (self.words_index, (spoken_language, signed_language, word)),
                (self.glosses_index, (spoken_language, signed_language, word)),
                (self.glosses_index, (spoken_language, signed_language, current_gloss)),
            ]

            for dict_index, (spoken_language, signed_language, term) in lookup_list:
                if spoken_language in dict_index:
                    if signed_language in dict_index[spoken_language]:
                        lower_term = term.lower()
                        if lower_term in dict_index[spoken_language][signed_language]:
                            rows = dict_index[spoken_language][signed_language][lower_term]
                            return self.get_pose(self.get_best_row(rows, term))

            if step_idx + 1 < len(preprocess_steps):
                next_step_fn = preprocess_steps[step_idx + 1]

        # For the backups: Normalize the word only if the gloss represents a standalone integer.
        # This avoids altering decimals or alphanumeric tokens (e.g. "3.14", "A3").
        # Examples: "3," → "3", "(42)" → "42"; NOT normalized: "3.14", "A3", "3rd"
        if should_normalize_integer_token(gloss):
            word = preprocess_steps[-2](word)

        # Backup strategy: revert to backup sign language
        if signed_language in LANGUAGE_BACKUP:
            return self.lookup(word, gloss, spoken_language, LANGUAGE_BACKUP[signed_language], source)

        # Backup strategy: revert to fingerspelling
        if self.backup is not None:
            return self.backup.lookup(word, gloss, spoken_language, signed_language, source)

        raise FileNotFoundError

    def lookup_sequence(self, glosses: Gloss, spoken_language: str, signed_language: str, source: str = None):
        def lookup_pair(pair):
            word, gloss = pair
            if word == "":
                return None

            try:
                return self.lookup(word, gloss, spoken_language, signed_language)
            except FileNotFoundError as e:
                print(e)
                return None

        with ThreadPoolExecutor() as executor:
            results = executor.map(lookup_pair, glosses)

        poses = [result for result in results if result is not None]  # Filter out None results

        if len(poses) == 0:
            gloss_sequence = ' '.join([f"{word}/{gloss}" for word, gloss in glosses])
            raise Exception(f"No poses found for {gloss_sequence}")

        return poses