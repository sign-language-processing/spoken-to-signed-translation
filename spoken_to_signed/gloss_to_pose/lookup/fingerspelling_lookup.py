from pathlib import Path

from pose_format import Pose

from .. import CSVPoseLookup, concatenate_poses


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

    def characters_lookup(self, word: str, spoken_language: str, signed_language: str):
        if word != "":
            rows = self.words_index[spoken_language][signed_language]
            alphabet = self.alphabets[spoken_language][signed_language]
            found = False
            for key in alphabet:
                if key in word:
                    found = True
                    match_index = word.index(key)

                    yield from self.characters_lookup(word[:match_index], spoken_language, signed_language)
                    yield self.get_pose(rows[key][0])
                    yield from self.characters_lookup(word[match_index + len(key):], spoken_language, signed_language)
                    break

            if not found:
                raise FileNotFoundError(f"Characters {word} not found in fingerspelling lexicon")

    def stretch_pose(self, pose: Pose, by: float) -> Pose:
        fps = pose.body.fps
        pose = pose.interpolate(fps * by)
        pose.body.fps = fps
        return pose

    def lookup(self, word: str, gloss: str, spoken_language: str, signed_language: str, source: str = None) -> Pose:
        if spoken_language not in self.words_index or signed_language not in self.words_index[spoken_language]:
            raise FileNotFoundError(
                f"Language pair {spoken_language} -> {signed_language} not supported for fingerspelling")

        poses = list(self.characters_lookup(word.lower(), spoken_language, signed_language))

        # hold the last letters longer to make it more readable
        poses[-1] = self.stretch_pose(poses[-1], 2)

        return concatenate_poses(poses)
