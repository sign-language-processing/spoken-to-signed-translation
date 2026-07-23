import math
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import NamedTuple, Optional

from pose_format import Pose

from spoken_to_signed.gloss_to_pose.languages import LANGUAGE_BACKUP
from spoken_to_signed.gloss_to_pose.lookup.lru_cache import LRUCache
from spoken_to_signed.text_to_gloss.types import Gloss


class CoverageType(str, Enum):
    # str subclass so values serialize to JSON as-is
    LEXICON = "lexicon"
    LANGUAGE_BACKUP = "language_backup"
    FINGERSPELLING_BACKUP = "fingerspelling_backup"
    UNMATCHED = "unmatched"


class TokenCoverage(NamedTuple):
    word: str
    gloss: str
    coverage: CoverageType


class PoseResult(NamedTuple):
    pose: Pose
    # Active-signing frame range within the pose, if known (e.g. from a precomputed
    # segmentation). None means "unknown -- fall back to the elbow heuristic".
    signing_span: Optional[tuple[int, int]] = None
    # How the lookup resolved; None for results that did not come from a lookup
    # (e.g. concatenated poses).
    coverage: Optional[CoverageType] = None


class PoseLookup:
    def __init__(self, rows: list, directory: str = None, backup: "PoseLookup" = None, cache: LRUCache = None):
        self.directory = directory

        self.words_index = self.make_dictionary_index(rows, based_on="words")
        self.glosses_index = self.make_dictionary_index(rows, based_on="glosses")

        self.backup = backup

        self.file_systems = {}
        self.cache = cache if cache is not None else LRUCache()

        # Per-token coverage of the most recent lookup_sequence call
        self.last_coverage: list[TokenCoverage] = []

    def make_dictionary_index(self, rows: list, based_on: str):
        # As an attempt to make the index more compact in memory, we store a dictionary with only what we need
        languages_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for d in rows:
            term = d[based_on]
            lower_term = term.lower()
            languages_dict[d["spoken_language"]][d["signed_language"]][lower_term].append(
                {
                    "path": d["path"],
                    "term": term,
                    "start": int(d["start"]),
                    "end": int(d["end"]),
                    # Optional precomputed active-signing bounds; default to the
                    # clip bounds, which the reader treats as "no segmentation".
                    "segment_start": int(d.get("segment_start") or d["start"]),
                    "segment_end": int(d.get("segment_end") or d["end"]),
                    "priority": int(d["priority"]),
                }
            )
        return languages_dict

    def read_pose(self, pose_path: str):
        if pose_path.startswith("gs://"):
            if "gcs" not in self.file_systems:
                import gcsfs

                self.file_systems["gcs"] = gcsfs.GCSFileSystem(anon=True)

            with self.file_systems["gcs"].open(pose_path, "rb") as f:
                return Pose.read(f.read())

        if pose_path.startswith("https://"):
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
        clip = Pose(pose.header, pose.body[start_frame:end_frame])

        # If a segmentation span was recorded (differs from the clip bounds),
        # return it in the clip's frame coordinates so trimming can use it instead
        # of the elbow heuristic; otherwise None (fall back to the heuristic).
        signing_span = None
        if (row["segment_start"], row["segment_end"]) != (row["start"], row["end"]):
            first = math.floor(row["segment_start"] // frame_time) - start_frame
            last = math.ceil(row["segment_end"] // frame_time) - start_frame
            signing_span = (max(0, first), min(len(clip.body.data), last))
        return clip, signing_span

    def get_best_row(self, rows, term: str):
        # Sort by priority: lower is "better"
        rows = sorted(rows, key=lambda x: x["priority"])
        # String match exact term
        for row in rows:
            if term == row["term"]:
                return row
        # Return the highest priority row
        return rows[0]

    def lookup(
        self, word: str, gloss: str, spoken_language: str, signed_language: str, source: str = None
    ) -> PoseResult:
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
                        pose, signing_span = self.get_pose(self.get_best_row(rows, term))
                        return PoseResult(pose=pose, signing_span=signing_span, coverage=CoverageType.LEXICON)

        # Backup strategy: revert to backup sign language
        if signed_language in LANGUAGE_BACKUP:
            result = self.lookup(word, gloss, spoken_language, LANGUAGE_BACKUP[signed_language], source)
            if result.coverage == CoverageType.LEXICON:
                result = result._replace(coverage=CoverageType.LANGUAGE_BACKUP)
            return result

        # Backup strategy: revert to fingerspelling
        if self.backup is not None:
            return self.backup.lookup(word, gloss, spoken_language, signed_language, source)

        raise FileNotFoundError

    def lookup_sequence(
        self, glosses: Gloss, spoken_language: str, signed_language: str, source: str = None
    ) -> list[PoseResult]:
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
            all_results = list(executor.map(lookup_pair, glosses))

        # Record how each token resolved, for optional coverage reporting
        self.last_coverage = [
            TokenCoverage(word, gloss, result.coverage if result is not None else CoverageType.UNMATCHED)
            for (word, gloss), result in zip(glosses, all_results)
            if word != ""
        ]

        results = [r for r in all_results if r is not None]

        if len(results) == 0:
            gloss_sequence = " ".join([f"{word}/{gloss}" for word, gloss in glosses])
            raise Exception(f"No poses found for {gloss_sequence}")

        return results
