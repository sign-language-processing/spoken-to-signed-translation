import argparse
import importlib

from pose_format import Pose

from spoken_to_signed.gloss_to_pose import PoseLookup, gloss_to_pose
from spoken_to_signed.text_to_gloss.types import Gloss


def _text_to_gloss(text: str, language: str, glosser: str) -> Gloss:
    module = importlib.import_module(f"spoken_to_signed.text_to_gloss.{glosser}")
    return module.text_to_gloss(text=text, language=language)


def _gloss_to_pose(gloss: Gloss, lexicon: str, spoken_language: str, signed_language: str) -> Pose:
    pose_lookup = PoseLookup(lexicon)
    return gloss_to_pose(gloss, pose_lookup, spoken_language, signed_language)


def _pose_to_video(pose: Pose, video_path: str):
    pass


def _text_input_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--glosser", choices=['simple', 'rules', 'nmt'], required=True)
    parser.add_argument("--spoken-language", choices=['de', 'fr', 'it', 'en'], required=True)
    parser.add_argument("--signed-language", choices=['sgg', 'gsg', 'bfi'], required=True)


def text_to_gloss():
    args_parser = argparse.ArgumentParser()
    _text_input_arguments(args_parser)
    args = args_parser.parse_args()

    print("Text to gloss")
    print("Input text:", args.text)
    gloss = _text_to_gloss(args.text, args.spoken_language, args.glosser)
    print("Output gloss:", gloss)


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

    gloss = _text_to_gloss(args.text, args.spoken_language, args.glosser)
    pose = _gloss_to_pose(gloss, args.lexicon, args.spoken_language, args.signed_language)

    with open(args.pose, "wb") as f:
        pose.write(f)

    print("Text to gloss to pose")
    print("Input text:", args.text)
    print("Output pose:", args.pose)


def text_to_gloss_to_pose_to_video():
    args_parser = argparse.ArgumentParser()
    _text_input_arguments(args_parser)
    args_parser.add_argument("--video", type=str, required=True)
    args = args_parser.parse_args()

    gloss = _text_to_gloss(args.text, args.spoken_language, args.glosser)
    pose = _gloss_to_pose(gloss, args.lexicon, args.spoken_language, args.signed_language)
    _pose_to_video(pose, args.video)

    print("Text to gloss to pose to video")
    print("Input text:", args.text)
    print("Output video:", args.video)
