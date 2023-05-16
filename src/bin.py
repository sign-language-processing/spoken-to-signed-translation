import argparse

from pose_format import Pose


def _text_to_gloss(text: str) -> str:
    return "TODO"


def _gloss_to_pose(gloss: str) -> Pose:
    pass


def _pose_to_video(pose: Pose, video_path: str):
    pass


def _text_input_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--glosser", choices=['simple', 'rules', 'nmt'], required=True)
    parser.add_argument("--spoken-language", choices=['de', 'fr', 'it', 'en'], required=True)
    parser.add_argument("--signed-language", choices=['ch', 'de', 'en'], required=True)


def text_to_gloss():
    args_parser = argparse.ArgumentParser()
    _text_input_arguments(args_parser)
    args = args_parser.parse_args()

    print("Text to gloss")
    print("Input text:", args.text)
    print("Output gloss:", _text_to_gloss(args.text))


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
    args_parser.add_argument("--pose", type=str, required=True)
    args = args_parser.parse_args()

    gloss = _text_to_gloss(args.text)
    pose = _gloss_to_pose(gloss)

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

    gloss = _text_to_gloss(args.text)
    pose = _gloss_to_pose(gloss)
    _pose_to_video(pose, args.video)

    print("Text to gloss to pose to video")
    print("Input text:", args.text)
    print("Output video:", args.video)
