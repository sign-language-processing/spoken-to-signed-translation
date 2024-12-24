import string
from pathlib import Path

from pose_format import Pose
from spoken_to_signed.gloss_to_pose import CSVPoseLookup, concatenate_poses


def stretch_pose(pose: Pose, by: float) -> Pose:
    fps = pose.body.fps
    pose = pose.interpolate(fps * by)
    pose.body.fps = fps
    return pose

class FingerspellingPoseLookup(CSVPoseLookup):
    def __init__(self):
        fs_directory = Path(__file__).parent.parent.parent / "assets" / "fingerspelling_lexicon"

        super().__init__(directory=str(fs_directory))

        # Precompute the sorted alphabets to make the lookup faster
        self.alphabets = {
            spoken_language: {
                signed_language: sorted(si_values.keys(), key=len, reverse=True)
                for signed_language, si_values in sp_values.items()
            }
            for spoken_language, sp_values in self.words_index.items()
        }

    def break_down_word(self, word: str, spoken_language:str, signed_language: str):
        if word != "":
            alphabet = self.alphabets[spoken_language][signed_language]
            found = False
            for key in alphabet:
                if key in word:
                    found = True
                    match_index = word.index(key)
                    yield from self.break_down_word(word[:match_index], spoken_language, signed_language)
                    yield key
                    yield from self.break_down_word(word[match_index + len(key):], spoken_language, signed_language)
                    break

            if not found:
                raise FileNotFoundError(f"Characters {word} not found in fingerspelling lexicon")

    def get_char_pose(self, char: str, spoken_language: str, signed_language: str) -> Pose:
        rows = self.words_index[spoken_language][signed_language]
        return self.get_pose(rows[char][0])

    def lookup(self, word: str, gloss: str, spoken_language: str, signed_language: str, source: str = None) -> Pose:
        if spoken_language not in self.words_index or signed_language not in self.words_index[spoken_language]:
            raise FileNotFoundError(
                f"Language pair {spoken_language} -> {signed_language} not supported for fingerspelling")

        word = self.break_down_word(word.lower(), spoken_language, signed_language)
        poses = [self.get_char_pose(char, spoken_language, signed_language) for char in word]

        # hold the last letters longer to make it more readable
        poses[-1] = stretch_pose(poses[-1], 2)

        return concatenate_poses(poses)

if __name__ == "__main__":
    lookup = FingerspellingPoseLookup()
    letters = string.ascii_lowercase
    pose = lookup.lookup(letters, letters, "en", "ase")
    with open("new.pose", "wb") as f:
        pose.write(f)
