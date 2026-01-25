import math
import os
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import List

from pose_format import Pose

from spoken_to_signed.gloss_to_pose.languages import LANGUAGE_BACKUP
from spoken_to_signed.gloss_to_pose.lookup.lru_cache import LRUCache
from spoken_to_signed.text_to_gloss.types import Gloss

_POSE_READ_LOCK = threading.Lock()


def preprocess_lower_strip(s: str) -> str:
    """Apply strip() and lower() exactly as normalize_token does at the beginning."""
    return str(s).strip().lower()

def preprocess_rule_based(s: str) -> str:
    """Remove '+' and trailing '-ix'."""
    s = s.replace("+", "")
    if s.endswith("-ix"):
        s = s[:-3]
    return s

def preprocess_spacylemma(s: str) -> str:
    """Remove '--' as done in cleanup_spacylemma."""
    return s.replace("--", "")

def preprocess_keep_letters_only(s: str) -> str:
    """Keep only Unicode letters."""
    return "".join(ch for ch in s if ch.isalpha())

def preprocess_integer_with_punctuation(s: str) -> str:
    """
    Normalize integer tokens ONLY when digits are not part of an alphanumeric word
    and are surrounded by non-letter characters.

    Keeps decimals and alphanumeric tokens untouched.
    """
    s = s.strip()

    # 1) Leave decimals untouched
    if re.fullmatch(r"\d+[.,]\d+", s):
        return s

    # 2) Reject alphanumeric tokens (letters touching digits)
    if re.search(r"[A-Za-z]\d|\d[A-Za-z]", s):
        return s

    # 3) Match pure integer with optional non-letter punctuation around it
    m = re.fullmatch(r"[^A-Za-z0-9]*([0-9]+)[^A-Za-z0-9]*", s)
    if m:
        return m.group(1)

    return s

def should_normalize_integer_token(s: str) -> bool:
    """
    Returns True ONLY when:
    - The token contains an integer
    - It is NOT a decimal
    - Digits are NOT adjacent to letters
    - Digits are surrounded by non-letter characters (or boundaries)
    """
    s = s.strip()

    # Has at least one digit
    if not re.search(r"\d", s):
        return False

    # Is a decimal → do NOT normalize
    if re.fullmatch(r"\d+[.,]\d+", s):
        return False

    # Alphanumeric (letters touching digits) → do NOT normalize
    if re.search(r"[A-Za-z]\d|\d[A-Za-z]", s):
        return False

    # Integer possibly wrapped by non-letter characters
    return bool(re.fullmatch(r"[^A-Za-z0-9]*\d+[^A-Za-z0-9]*", s))


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
                "start": float(d['start']),
                "end": float(d['end']),
                "priority": int(d['priority']),
            })
        return languages_dict

    def read_pose(self, pose_path: str):
        """Safely read a pose file, avoiding concurrent Pose.read issues."""
        if pose_path.startswith('gs://'):
            if 'gcs' not in self.file_systems:
                import gcsfs
                self.file_systems['gcs'] = gcsfs.GCSFileSystem(anon=True)
            with self.file_systems['gcs'].open(pose_path, "rb") as f:
                data = f.read()

        elif pose_path.startswith('https://'):
            raise NotImplementedError("Can't access pose files from https endpoint")

        else:
            if self.directory is None:
                raise ValueError("Can't access pose files without specifying a directory")
            full_path = os.path.join(self.directory, pose_path)
            with open(full_path, "rb") as f:
                data = f.read()

        # Serialize the actual Pose.read (thread-unsafe part)
        with _POSE_READ_LOCK:
            try:
                return Pose.read(data)
            except TypeError as e:
                # Add extra context for debugging corrupted/truncated files
                raise RuntimeError(
                    f"Failed to decode pose file {pose_path} "
                    f"(size={len(data)} bytes)"
                ) from e

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

    def lookup(self, word: str, gloss: str, spoken_language: str, signed_language: str, source: str = None, verbose: bool = False) -> Pose:

        def log(msg: str):
            if verbose:
                print(msg)

        # List of progressive transformations applied ONLY in the main language
        preprocess_steps = [
            None,                           # original attempt
            preprocess_lower_strip,
            preprocess_rule_based,
            preprocess_spacylemma,
            preprocess_integer_with_punctuation,
            preprocess_keep_letters_only
        ]

        original_gloss = gloss
        current_gloss = gloss

        # Attempts within the main language
        for step_idx, step_fn in enumerate(preprocess_steps):

            # Apply cumulative transformations except for attempt #0
            if step_fn is not None:
                current_gloss = step_fn(current_gloss)

            log(f"[GLOSS '{original_gloss}'] [STEP #{step_idx}] Trying lookup with current_gloss='{current_gloss}'")

            lookup_list = [
                (self.words_index,   (spoken_language, signed_language, word)),
                (self.glosses_index, (spoken_language, signed_language, word)),
                (self.glosses_index, (spoken_language, signed_language, current_gloss)),
            ]

            for dict_index, (sp_lang, sg_lang, term) in lookup_list:
                if sp_lang in dict_index:
                    if sg_lang in dict_index[sp_lang]:
                        lower_term = term.lower()
                        if lower_term in dict_index[sp_lang][sg_lang]:
                            rows = dict_index[sp_lang][sg_lang][lower_term]
                            log(
                                f"[SUCCESS] Found entry for term='{term}' (lower='{lower_term}') "
                                f"with gloss_version='{current_gloss}'\n"
                            )
                            return self.get_pose(self.get_best_row(rows, term))

            # If we reached here, this step has failed
            next_msg = ""

            if step_idx + 1 < len(preprocess_steps):
                next_step_fn = preprocess_steps[step_idx + 1]
                if next_step_fn is not None:
                    preview_gloss = next_step_fn(current_gloss)
                    next_msg = f" Trying next gloss version: '{preview_gloss}'."
                else:
                    next_msg = " Trying next lookup configuration."
            else:
                next_msg = " No more preprocessing steps. Will try backup strategies (if available)."

            log(
                f"[GLOSS '{original_gloss}'] [FAIL #{step_idx}] "
                f"Lookup failed for gloss='{current_gloss}'.{next_msg}"
            )

        # =============================
        # BACKUP STRATEGIES (USE ORIGINAL GLOSS)
        # =============================

        # 1) Backup to secondary language using ORIGINAL gloss (not the transformed one)
        if signed_language in LANGUAGE_BACKUP:
            if should_normalize_integer_token(original_gloss):
                word = preprocess_integer_with_punctuation(word)
                log(
                    f"[GLOSS '{original_gloss}'] Attempting with secondary language backup, "
                    f"using '{word}' as word."
                )
            else:
                log(f"[GLOSS '{original_gloss}'] Attempting with secondary language backup")

            return self.lookup(word, original_gloss, spoken_language, LANGUAGE_BACKUP[signed_language], source, verbose=verbose)

        # 2) Backup to fingerspelling using ORIGINAL gloss
        if self.backup is not None:
            if should_normalize_integer_token(original_gloss):
                word = preprocess_integer_with_punctuation(word)
                log(
                    f"[GLOSS '{original_gloss}'] Attempting with fingerspelling backup, "
                    f"using '{word}' as word."
                )
            else:
                log(f"[GLOSS '{original_gloss}'] Attempting with fingerspelling backup")

            return self.backup.lookup(word, original_gloss, spoken_language, signed_language, source, verbose=verbose)

        # If everything fails:
        raise FileNotFoundError(
            f"Could not resolve word='{word}' gloss='{gloss}' even after preprocessing and backups."
        )

    def lookup_sequence(self, glosses: Gloss, spoken_language: str, signed_language: str, source: str = None, coverage_info: bool = False):
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
            results = list(executor.map(lookup_pair, glosses))

        poses = [result for result in results if result is not None]  # Filter out None results

        if len(poses) == 0:
            gloss_sequence = ' '.join([f"{word}/{gloss}" for word, gloss in glosses])
            raise Exception(f"No poses found for {gloss_sequence}")
        
        if coverage_info:
            total = len(results)
            success = len(poses)
            coverage = f"{success / total:.3f}" if total > 0 else "0.000"
            return poses, coverage

        return poses
