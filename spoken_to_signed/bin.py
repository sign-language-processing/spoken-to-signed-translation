import argparse
import csv
import importlib
import os
import tempfile
from itertools import chain

import numpy as np
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody

from spoken_to_signed.gloss_to_pose import (
    CSVPoseLookup,
    GlossToPoseResult,
    concatenate_poses,
    gloss_to_pose,
)
from spoken_to_signed.gloss_to_pose.coverage import CoverageStats, TokenCoverage
from spoken_to_signed.gloss_to_pose.lookup.fingerspelling_lookup import (
    FingerspellingPoseLookup,
)
from spoken_to_signed.text_to_gloss.types import Gloss


def _text_to_gloss(text: str, language: str, glosser: str, **kwargs) -> list[Gloss]:
    module = importlib.import_module(f"spoken_to_signed.text_to_gloss.{glosser}")
    return module.text_to_gloss(text=text, language=language, **kwargs)


def _gloss_to_pose(
    sentences: list[Gloss],
    lexicon: str,
    spoken_language: str,
    signed_language: str,
    coverage_info: bool = False,
    use_fingerspelling: bool = True,
) -> GlossToPoseResult:
    fingerspelling_lookup = FingerspellingPoseLookup() if use_fingerspelling else None
    pose_lookup = CSVPoseLookup(lexicon, backup=fingerspelling_lookup)
    results = [
        gloss_to_pose(gloss, pose_lookup, spoken_language, signed_language, coverage_info=coverage_info)
        for gloss in sentences
    ]

    poses = [r.pose for r in results]
    pose = poses[0] if len(poses) == 1 else concatenate_poses(poses, trim=False)

    if coverage_info:
        all_token_coverages = [r.token_coverages for r in results]
        return GlossToPoseResult(pose=pose, token_coverages=all_token_coverages)

    return GlossToPoseResult(pose=pose)


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


def _print_token_coverage(all_token_coverages: list[list[TokenCoverage]]):
    total = sum(len(s) for s in all_token_coverages)
    matched = sum(tc.exact_lexicon_match for s in all_token_coverages for tc in s)
    for sentence_coverages in all_token_coverages:
        for tc in sentence_coverages:
            status = "✓" if tc.exact_lexicon_match else "✗"
            print(f"  {status}  {tc.word} -> {tc.gloss}")
    print(f"Coverage: {matched / total:.3f} ({matched}/{total} tokens matched)")


def text_to_gloss_to_pose():
    args_parser = argparse.ArgumentParser()
    _text_input_arguments(args_parser)
    args_parser.add_argument("--lexicon", type=str, required=True)
    args_parser.add_argument(
        "--coverage-info",
        action="store_true",
        help="Print per-token gloss coverage to stdout.",
    )
    args_parser.add_argument(
        "--coverage-stats",
        type=str,
        default=None,
        help="Path to save per-token coverage statistics as a JSON file.",
    )
    args_parser.add_argument("--pose", type=str, required=True)
    args_parser.add_argument(
        "--no-fingerspelling",
        action="store_true",
        help="Disable fingerspelling fallback during pose lookup.",
    )
    args = args_parser.parse_args()

    need_coverage = args.coverage_info or args.coverage_stats is not None

    sentences = _text_to_gloss(args.text, args.spoken_language, args.glosser)
    result = _gloss_to_pose(
        sentences, args.lexicon, args.spoken_language, args.signed_language, need_coverage,
        use_fingerspelling=not args.no_fingerspelling
    )

    print("Text to gloss to pose")
    print("Input text:", args.text)
    print("Output pose:", args.pose)

    if need_coverage:
        _print_token_coverage(result.token_coverages)
        if args.coverage_stats:
            stats = CoverageStats()
            for sentence_coverages in result.token_coverages:
                stats.add_sentence(sentence_coverages, text=args.text)
            stats.save(args.coverage_stats)
            print(f"Coverage stats saved to: {args.coverage_stats}")

    with open(args.pose, "wb") as f:
        result.pose.write(f)


def _raw_concatenate_poses(poses: list[Pose]) -> Pose:
    new_data = np.concatenate([pose.body.data for pose in poses])
    new_conf = np.concatenate([pose.body.confidence for pose in poses])
    new_body = NumPyPoseBody(fps=poses[0].body.fps, data=new_data, confidence=new_conf)
    return Pose(header=poses[0].header, body=new_body)


def _write_chunk(chunk_poses: list[Pose], chunk_path: str):
    chunk_pose = _raw_concatenate_poses(chunk_poses) if len(chunk_poses) > 1 else chunk_poses[0]
    with open(chunk_path, "wb") as f:
        chunk_pose.write(f)


def _process_pose(text, spoken_language, glosser, pose_lookup, signed_language, need_coverage, stats):
    sentences = _text_to_gloss(text, spoken_language, glosser)
    results = [
        gloss_to_pose(gloss, pose_lookup, spoken_language, signed_language, coverage_info=need_coverage)
        for gloss in sentences
    ]
    poses = [r.pose for r in results]
    if need_coverage:
        for r in results:
            stats.add_sentence(r.token_coverages, text=text)
    return poses[0] if len(poses) == 1 else concatenate_poses(poses, trim=False)


def _bulk_sequential(texts, args, pose_lookup, need_coverage, stats):
    for i, text in enumerate(texts):
        pose = _process_pose(
            text, args.spoken_language, args.glosser, pose_lookup, args.signed_language, need_coverage, stats
        )
        pose_path = os.path.join(args.output_dir, f"{i:06d}.pose")
        with open(pose_path, "wb") as f:
            pose.write(f)
        print(f"[{i + 1}/{len(texts)}] {pose_path}")


def _bulk_compacted(texts, args, pose_lookup, need_coverage, stats):
    metadata_rows: list[dict] = []
    chunk_index = 0
    chunk_poses: list[Pose] = []
    chunk_frame_count = 0

    for i, text in enumerate(texts):
        pose = _process_pose(
            text, args.spoken_language, args.glosser, pose_lookup, args.signed_language, need_coverage, stats
        )
        pose_frames = len(pose.body.data)

        # Flush current chunk if adding this pose would exceed the limit (keep at least one pose per chunk)
        if chunk_poses and chunk_frame_count + pose_frames > args.max_frames_per_chunk:
            chunk_path = os.path.join(args.output_dir, f"chunk_{chunk_index:06d}.pose")
            _write_chunk(chunk_poses, chunk_path)
            print(f"  Saved chunk {chunk_index}: {chunk_path} ({chunk_frame_count} frames)")
            chunk_index += 1
            chunk_poses = []
            chunk_frame_count = 0

        start_frame = chunk_frame_count
        end_frame = chunk_frame_count + pose_frames - 1
        chunk_path = os.path.join(args.output_dir, f"chunk_{chunk_index:06d}.pose")
        metadata_rows.append(
            {"text": text, "pose_file": os.path.abspath(chunk_path), "start_frame": start_frame, "end_frame": end_frame}
        )
        chunk_poses.append(pose)
        chunk_frame_count += pose_frames
        print(f"[{i + 1}/{len(texts)}] buffered into {chunk_path} frames {start_frame}–{end_frame}")

    # Write the last chunk
    if chunk_poses:
        chunk_path = os.path.join(args.output_dir, f"chunk_{chunk_index:06d}.pose")
        _write_chunk(chunk_poses, chunk_path)
        print(f"  Saved chunk {chunk_index}: {chunk_path} ({chunk_frame_count} frames)")

    # Write metadata TSV
    metadata_path = os.path.join(args.output_dir, "metadata.tsv")
    with open(metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "pose_file", "start_frame", "end_frame"], delimiter="\t")
        writer.writeheader()
        writer.writerows(metadata_rows)
    print(f"Metadata saved to: {metadata_path}")


def text_to_gloss_to_pose_bulk():
    args_parser = argparse.ArgumentParser(
        description="Translate a file of texts (one per line) into pose files in bulk."
    )
    args_parser.add_argument("--texts", type=str, required=True, help="Path to a text file with one sentence per line.")
    args_parser.add_argument("--glosser", choices=["simple", "spacylemma", "rules", "nmt"], required=True)
    args_parser.add_argument("--lexicon", type=str, required=True)
    args_parser.add_argument("--spoken-language", type=str, required=True)
    args_parser.add_argument("--signed-language", type=str, required=True)
    args_parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where output .pose files are written (named 000000.pose, 000001.pose, …).",
    )
    args_parser.add_argument(
        "--coverage-stats",
        type=str,
        default=None,
        help="Path to save aggregated per-token coverage statistics as a JSON file.",
    )
    args_parser.add_argument(
        "--compacted-poses",
        action="store_true",
        help="Concatenate generated poses into chunks instead of saving one file per sentence.",
    )
    args_parser.add_argument(
        "--max-frames-per-chunk",
        type=int,
        default=10000,
        help="Maximum number of frames per chunk when --compacted-poses is set (default: 10000).",
    )
    args_parser.add_argument(
        "--no-fingerspelling",
        action="store_true",
        help="Disable fingerspelling fallback during pose lookup.",
    )
    args = args_parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.texts, encoding="utf-8") as f:
        texts = [line.rstrip("\n") for line in f if line.strip()]

    need_coverage = args.coverage_stats is not None
    stats = CoverageStats() if need_coverage else None
    fingerspelling_lookup = FingerspellingPoseLookup() if not args.no_fingerspelling else None
    pose_lookup = CSVPoseLookup(args.lexicon, backup=fingerspelling_lookup)

    if args.compacted_poses:
        _bulk_compacted(texts, args, pose_lookup, need_coverage, stats)
    else:
        _bulk_sequential(texts, args, pose_lookup, need_coverage, stats)

    if need_coverage:
        stats.save(args.coverage_stats)
        print(f"Coverage stats saved to: {args.coverage_stats}")
        print(f"Overall coverage: {stats.fraction:.3f} ({stats.matched_tokens}/{stats.total_tokens} tokens matched)")


def text_to_gloss_to_pose_to_video():
    args_parser = argparse.ArgumentParser()
    _text_input_arguments(args_parser)
    args_parser.add_argument("--lexicon", type=str, required=True)
    args_parser.add_argument("--video", type=str, required=True)
    args_parser.add_argument(
        "--no-fingerspelling",
        action="store_true",
        help="Disable fingerspelling fallback during pose lookup.",
    )
    args = args_parser.parse_args()

    sentences = _text_to_gloss(args.text, args.spoken_language, args.glosser, signed_language=args.signed_language)
    result = _gloss_to_pose(sentences, args.lexicon, args.spoken_language, args.signed_language,
                            use_fingerspelling=not args.no_fingerspelling)
    _pose_to_video(result.pose, args.video)

    print("Text to gloss to pose to video")
    print("Input text:", args.text)
    print("Output video:", args.video)


if __name__ == "__main__":
    text_to_gloss_to_pose()
