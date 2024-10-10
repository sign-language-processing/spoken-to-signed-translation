import argparse
import importlib
import os
import tempfile
from typing import List

from pose_format import Pose

from spoken_to_signed.gloss_to_pose import gloss_to_pose, CSVPoseLookup, concatenate_poses
from spoken_to_signed.gloss_to_pose.lookup.fingerspelling_lookup import FingerspellingPoseLookup
from spoken_to_signed.text_to_gloss.types import Gloss


def _text_to_gloss(text: str, language: str, glosser: str) -> List[Gloss]:
    module = importlib.import_module(f"spoken_to_signed.text_to_gloss.{glosser}")
    return module.text_to_gloss(text=text, language=language)


def _gloss_to_pose(sentences: List[Gloss], lexicon: str, spoken_language: str, signed_language: str) -> Pose:
    fingerspelling_lookup = FingerspellingPoseLookup()
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
        print("Downloading pix2pix model")
        import urllib.request
        urllib.request.urlretrieve(
            "https://firebasestorage.googleapis.com/v0/b/sign-mt-assets/o/models%2Fgenerator%2Fmodel.h5?alt=media",
            pix2pix_path)

    import subprocess

    try:
        subprocess.run(["command", "-v", "pose_to_video"], shell=True, check=True)
    except subprocess.CalledProcessError:
        raise RuntimeError(
            "The command 'pose_to_video' does not exist. Please install the `transcription` package using `pip install git+https://github.com/sign-language-processing/transcription`")

    pose_path = tempfile.mktemp(suffix=".pose")
    with open(pose_path, "wb") as f:
        pose.write(f)

    args = ["pose_to_video", "--type=pix_to_pix",
            "--model", pix2pix_path,
            "--pose", pose_path,
            "--video", video_path,
            "--upscale"]
    print(" ".join(args))
    subprocess.run(args, shell=True, check=True)


def _text_input_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--glosser", choices=['simple', 'spacylemma', 'rules', 'nmt'], required=True)
    parser.add_argument("--spoken-language", choices=['de', 'fr', 'it', 'en'], required=True)
    parser.add_argument("--signed-language", choices=['sgg', 'gsg', 'bfi', 'ase'], required=True)


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
    args = args_parser.parse_args()

    sentences = _text_to_gloss(args.text, args.spoken_language, args.glosser)
    pose = _gloss_to_pose(sentences, args.lexicon, args.spoken_language, args.signed_language)

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
    args = args_parser.parse_args()

    sentences = _text_to_gloss(args.text, args.spoken_language, args.glosser)
    pose = _gloss_to_pose(sentences, args.lexicon, args.spoken_language, args.signed_language)
    _pose_to_video(pose, args.video)

    print("Text to gloss to pose to video")
    print("Input text:", args.text)
    print("Output video:", args.video)

if __name__ == "__main__":
    text_to_gloss_to_pose()