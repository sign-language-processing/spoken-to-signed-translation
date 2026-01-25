import argparse
import importlib
import os
import tempfile
from pathlib import Path
from itertools import chain
from typing import List, Tuple, Union

from pose_format import Pose

from spoken_to_signed.gloss_to_pose import gloss_to_pose, CSVPoseLookup, concatenate_poses
from spoken_to_signed.gloss_to_pose.lookup.fingerspelling_lookup import FingerspellingPoseLookup
from spoken_to_signed.text_to_gloss.types import Gloss


def _text_to_gloss(text: str, language: str, glosser: str, **kwargs) -> List[Gloss]:
    module = importlib.import_module(f"spoken_to_signed.text_to_gloss.{glosser}")
    return module.text_to_gloss(text=text, language=language, **kwargs)


def add_coverage_to_pose_path(pose_path: str, coverage: str) -> str:
    """
    Append coverage info to a pose filename.

    Example:
        /path/sample.pose + 0.900 → /path/sample_cov0_900.pose
    """
    path = Path(pose_path)
    coverage_token = coverage.replace(".", "_")
    return str(path.with_name(f"{path.stem}_cov{coverage_token}{path.suffix}"))


def _gloss_to_pose(sentences: List[Gloss], lexicon: str, spoken_language: str, signed_language: str, coverage_info: bool = False) -> Union[Pose, Tuple[Pose, str]]:
    fingerspelling_lookup = FingerspellingPoseLookup()
    pose_lookup = CSVPoseLookup(lexicon, backup=fingerspelling_lookup)

    results = [
        gloss_to_pose(gloss, pose_lookup, spoken_language, signed_language, coverage_info=coverage_info)
        for gloss in sentences
    ]

    # --- Backward-compatible path (original behavior) ---
    if not coverage_info:
        poses = results # gloss_to_pose returns Pose
        return poses[0] if len(poses) == 1 else concatenate_poses(poses, trim=False)

    # --- Coverage-aware path ---
    poses = [pose for pose, _ in results]
    coverages = [float(c) for _, c in results]
    min_coverage = f"{min(coverages):.3f}"

    return (
        poses[0] if len(poses) == 1 else concatenate_poses(poses, trim=False),
        min_coverage,
    )


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
            "The command 'pose_to_video' does not exist. Please install the `transcription` package using "
            "`pip install git+https://github.com/sign-language-processing/transcription`")

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

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--lexicon", type=str)
    pre_args, _ = pre_parser.parse_known_args()

    if pre_args.lexicon:
        lookup = CSVPoseLookup(pre_args.lexicon)
        spoken_languages = list(lookup.words_index.keys())
        signed_languages = set(chain.from_iterable(lookup.words_index[lang].keys() for lang in spoken_languages))
    else:
        spoken_languages = ['de', 'fr', 'it', 'en']
        signed_languages = ['sgg', 'gsg', 'bfi', 'ase']

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
    args_parser.add_argument("--coverage-info", action="store_true",
                             help="Enables gloss coverage computation and reporting."
    )
    args_parser.add_argument("--pose", type=str, required=True)
    args = args_parser.parse_args()

    sentences = _text_to_gloss(args.text, args.spoken_language, args.glosser)

    result = _gloss_to_pose(sentences, args.lexicon, args.spoken_language, args.signed_language, args.coverage_info)

    print("Text to gloss to pose")
    print("Input text:", args.text)
    if args.coverage_info:
        pose, coverage = result
        output_path = add_coverage_to_pose_path(args.pose, coverage)
        print(f"Output pose: (coverage: {coverage}): {output_path}")
    else:
        pose = result
        output_path = args.pose
        print("Output pose:", args.pose)
    
    with open(output_path, "wb") as f:
        pose.write(f)


def text_to_gloss_to_pose_to_video():
    args_parser = argparse.ArgumentParser()
    _text_input_arguments(args_parser)
    args_parser.add_argument("--lexicon", type=str, required=True)
    args_parser.add_argument("--video", type=str, required=True)
    args = args_parser.parse_args()

    sentences = _text_to_gloss(args.text, args.spoken_language, args.glosser, signed_language=args.signed_language)
    pose = _gloss_to_pose(sentences, args.lexicon, args.spoken_language, args.signed_language)
    _pose_to_video(pose, args.video)

    print("Text to gloss to pose to video")
    print("Input text:", args.text)
    print("Output video:", args.video)


if __name__ == "__main__":
    text_to_gloss_to_pose()
