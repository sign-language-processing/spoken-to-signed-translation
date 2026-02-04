import argparse
import importlib
import os
import tempfile
from itertools import chain

from pose_format import Pose

from spoken_to_signed.gloss_to_pose import (
    CSVPoseLookup,
    concatenate_poses,
    gloss_to_pose,
)
from spoken_to_signed.gloss_to_pose.lookup.fingerspelling_lookup import (
    FingerspellingPoseLookup,
)
from spoken_to_signed.text_to_gloss.types import Gloss


def _text_to_gloss(text: str, language: str, glosser: str, **kwargs) -> list[Gloss]:
    module = importlib.import_module(f"spoken_to_signed.text_to_gloss.{glosser}")
    return module.text_to_gloss(text=text, language=language, **kwargs)


def _gloss_to_pose(sentences: list[Gloss], lexicon: str, spoken_language: str, signed_language: str, use_fingerspelling: bool = True) -> Pose:
    fingerspelling_lookup = FingerspellingPoseLookup() if use_fingerspelling else None
    pose_lookup = CSVPoseLookup(lexicon, backup=fingerspelling_lookup)
    poses = [gloss_to_pose(gloss, pose_lookup, spoken_language, signed_language) for gloss in sentences]
    if len(poses) == 1:
        return poses[0]
    return concatenate_poses(poses, trim=False)


def _get_models_dir():
    home_dir = os.path.expanduser("~")
    sign_dir = os.path.join(home_dir, ".sign")
    os.makedirs(sign_dir, exist_ok=True)
    models_dir = os.path.join(sign_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def _pose_to_video(pose: Pose, video_path: str):
    models_dir = _get_models_dir()
    pix2pix_path = os.path.join(models_dir, "pix2pix.h5")
    if not os.path.exists(pix2pix_path):
        print("Downloading pix2pix model...")
        import urllib.request

        urllib.request.urlretrieve(
            "https://firebasestorage.googleapis.com/v0/b/sign-mt-assets/o/models%2Fgenerator%2Fmodel.h5?alt=media",
            pix2pix_path,
        )

    import shutil
    import subprocess

    if shutil.which("pose_to_video") is None:
        raise RuntimeError(
            "The command 'pose_to_video' does not exist. Please install the `pose-to-video` package using "
            "`pip install 'pose-to-video[pix2pix,simple_upscaler] @ git+https://github.com/sign-language-processing/pose-to-video'`"
        )

    pose_path = tempfile.mktemp(suffix=".pose")
    with open(pose_path, "wb") as f:
        pose.write(f)

    args = [
        "pose_to_video",
        "--type=pix2pix",
        "--model",
        pix2pix_path,
        "--pose",
        pose_path,
        "--video",
        video_path,
        "--processors",
        "simple_upscaler",
    ]
    print(" ".join(args))
    subprocess.run(args, check=True)


def _text_input_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--glosser", choices=["simple", "spacylemma", "rules", "nmt"], required=True)

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--lexicon", type=str)
    pre_args, _ = pre_parser.parse_known_args()

    if pre_args.lexicon:
        lookup = CSVPoseLookup(pre_args.lexicon)
        spoken_languages = list(lookup.words_index.keys())
        signed_languages = set(chain.from_iterable(lookup.words_index[lang].keys() for lang in spoken_languages))
    else:
        spoken_languages = ["de", "fr", "it", "en"]
        signed_languages = ["sgg", "gsg", "bfi", "ase"]

    parser.add_argument("--spoken-language", choices=spoken_languages, required=True)
    parser.add_argument("--signed-language", choices=signed_languages, required=True)


def text_to_gloss():
    args_parser = argparse.ArgumentParser()
    _text_input_arguments(args_parser)
    args = args_parser.parse_args()

    print("Text to gloss")
    print("Input text:", args.text)
    sentences = _text_to_gloss(args.text, args.spoken_language, args.glosser)
    print("Output gloss:", sentences)


def pose_to_video():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--pose", type=str, required=True)
    args_parser.add_argument("--video", type=str, required=True)
    args = args_parser.parse_args()

    with open(args.pose, "rb") as f:
        pose = Pose.read(f.read())

    _pose_to_video(pose, args.video)

    print("Pose to video")
    print("Input pose:", args.pose)
    print("Output video:", args.video)


def text_to_gloss_to_pose():
    args_parser = argparse.ArgumentParser()
    _text_input_arguments(args_parser)
    args_parser.add_argument("--lexicon", type=str, required=True)
    args_parser.add_argument("--pose", type=str, required=True)
    args_parser.add_argument(
        "--no-fingerspelling",
        action="store_true",
        help="Disable fingerspelling fallback during pose lookup",
    )
    args = args_parser.parse_args()
    use_fingerspelling = not args.no_fingerspelling

    sentences = _text_to_gloss(args.text, args.spoken_language, args.glosser)
    pose = _gloss_to_pose(sentences, args.lexicon, args.spoken_language, args.signed_language, use_fingerspelling)

    with open(args.pose, "wb") as f:
        pose.write(f)

    print("Text to gloss to pose")
    print("Input text:", args.text)
    print("Output pose:", args.pose)


def text_to_gloss_to_pose_to_video():
    args_parser = argparse.ArgumentParser()
    _text_input_arguments(args_parser)
    args_parser.add_argument("--lexicon", type=str, required=True)
    args_parser.add_argument("--video", type=str, required=True)
    args_parser.add_argument(
        "--no-fingerspelling",
        action="store_true",
        help="Disable fingerspelling fallback during pose lookup",
    )
    args = args_parser.parse_args()
    use_fingerspelling = not args.no_fingerspelling

    sentences = _text_to_gloss(args.text, args.spoken_language, args.glosser, signed_language=args.signed_language)
    pose = _gloss_to_pose(sentences, args.lexicon, args.spoken_language, args.signed_language, use_fingerspelling)
    _pose_to_video(pose, args.video)

    print("Text to gloss to pose to video")
    print("Input text:", args.text)
    print("Output video:", args.video)


if __name__ == "__main__":
    text_to_gloss_to_pose()
