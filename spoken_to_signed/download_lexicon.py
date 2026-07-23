import argparse
import csv
import os
from datetime import datetime
from typing import Optional

from pose_format import Pose, PoseHeader
from pose_format.numpy import NumPyPoseBody
from pose_format.utils.reader import BufferReader
from tqdm import tqdm

# segment_start/segment_end optionally hold precomputed active-signing bounds (e.g.
# from a segmentation model); left equal to start/end, the reader falls back to its
# elbow heuristic, so populating them is a build-time-only concern.
LEXICON_INDEX = [
    "path",
    "spoken_language",
    "signed_language",
    "start",
    "end",
    "segment_start",
    "segment_end",
    "words",
    "glosses",
    "priority",
]


def init_index(index_path: str):
    if not os.path.isfile(index_path):
        # Create csv file with specified header
        with open(index_path, "w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(LEXICON_INDEX)


def _segment_loader():
    """Load the sign-segmentation model used to precompute active-signing bounds,
    or None to skip segmentation (the reader then falls back to its elbow
    heuristic). Enabled by setting SEGMENTATION_MODEL_DIR to a local model
    directory (a safetensors dir or a Lightning .ckpt); this keeps the model a
    build-time-only, opt-in dependency -- it never touches the translation runtime.
    """
    model_dir = os.environ.get("SEGMENTATION_MODEL_DIR")
    if not model_dir:
        return None
    try:
        from sign_language_segmentation.inference.adapters.model_store import ModelStore
    except ImportError:
        print(
            "SEGMENTATION_MODEL_DIR is set but sign_language_segmentation is not installed; "
            "skipping segmentation. Install it to precompute segment bounds."
        )
        return None
    return ModelStore(model_dir=model_dir, device="cpu")


def _segment_span_ms(pose: Pose, loader) -> Optional[tuple[int, int]]:
    """The [start, end] of the active signing in milliseconds from the segmentation
    model, or None if unavailable / no sign detected."""
    if loader is None:
        return None
    from sign_language_segmentation.inference.core.segmentation import segment_pose

    out = segment_pose(pose, model_loader=loader, device="cpu")
    tiers = out[1] if isinstance(out, tuple) else out
    signs = tiers.get("SIGN", [])
    if not signs:
        return None
    fps = pose.body.fps
    first = min(s["start"] for s in signs)
    last = max(s["end"] for s in signs)
    return round(first * 1000 / fps), round(last * 1000 / fps)


def load_signsuisse(directory_path: str) -> list[dict[str, str]]:
    try:
        import sign_language_datasets  # noqa: F401
    except ImportError as e:
        raise ImportError("Please install sign_language_datasets. pip install sign-language-datasets") from e

    # noinspection PyUnresolvedReferences
    import sign_language_datasets.datasets.signsuisse as signsuisse  # noqa: F401
    import tensorflow_datasets as tfds
    from sign_language_datasets.datasets.config import SignDatasetConfig

    # noinspection PyUnresolvedReferences
    from sign_language_datasets.datasets.signsuisse.signsuisse import _POSE_HEADERS

    iana_tags = {
        "ch-de": "sgg",
        "ch-fr": "ssr",
        "ch-it": "slf",
    }

    # for cache busting, we use today's date
    date_str = datetime.now().strftime("%Y-%m-%d")
    config = SignDatasetConfig(name=date_str, version="1.0.0", include_video=False, include_pose="holistic")
    dataset = tfds.load(name="sign_suisse", builder_kwargs={"config": config})

    with open(_POSE_HEADERS["holistic"], "rb") as buffer:
        pose_header = PoseHeader.read(BufferReader(buffer.read()))

    segmenter = _segment_loader()

    for datum in tqdm(dataset["train"]):
        uid_raw = datum["id"].numpy().decode("utf-8")
        spoken_language = datum["spokenLanguage"].numpy().decode("utf-8")
        signed_language = iana_tags[datum["signedLanguage"].numpy().decode("utf-8")]
        words = datum["name"].numpy().decode("utf-8")

        # Load pose and save to file
        tf_pose = datum["pose"]
        fps = int(tf_pose["fps"].numpy())
        if fps == 0:
            continue
        pose_body = NumPyPoseBody(fps, tf_pose["data"].numpy(), tf_pose["conf"].numpy())
        pose = Pose(pose_header, pose_body)
        pose_relative_path = os.path.join(signed_language, f"{uid_raw}.pose")
        os.makedirs(os.path.join(directory_path, signed_language), exist_ok=True)
        with open(os.path.join(directory_path, pose_relative_path), "wb") as f:
            pose.write(f)

        # Timings are integer milliseconds (what the reader expects). Segment bounds
        # come from the model when enabled; otherwise they equal the clip so the
        # reader falls back to its elbow heuristic.
        duration_ms = round(1000 * len(pose_body.data) / fps)
        span = _segment_span_ms(pose, segmenter)
        segment_start, segment_end = span if span is not None else (0, duration_ms)

        yield {
            "path": pose_relative_path,
            "spoken_language": spoken_language,
            "signed_language": signed_language,
            "words": words,
            "start": "0",
            "end": str(duration_ms),
            "segment_start": str(segment_start),
            "segment_end": str(segment_end),
            "glosses": "",
            "priority": "0",
        }


def normalize_row(row: dict[str, str]):
    if row["glosses"] == "" and row["words"] != "":
        from spoken_to_signed.text_to_gloss.simple import text_to_gloss

        try:
            sentences = text_to_gloss(text=row["words"], language=row["spoken_language"])
            glosses = [g for sentence in sentences for w, g in sentence]
            row["glosses"] = " ".join(glosses)
        except ValueError as e:
            if not ("Language" in str(e) and "not supported" in str(e)):
                raise e


def get_data(name: str, directory: str):
    data_loaders = {
        "signsuisse": load_signsuisse,
    }
    if name not in data_loaders:
        raise NotImplementedError(f"{name} is unknown.")

    return data_loaders[name](directory)


def add_data(data: list[dict[str, str]], directory: str):
    index_path = os.path.join(directory, "index.csv")
    os.makedirs(directory, exist_ok=True)
    init_index(index_path)

    with open(index_path, "a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        for row in tqdm(data):
            normalize_row(row)
            writer.writerow([row[key] for key in LEXICON_INDEX])

    print(f"Added entries to {index_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", choices=["signsuisse"], required=True)
    parser.add_argument("--directory", type=str, required=True)
    args = parser.parse_args()

    data = get_data(args.name, args.directory)
    add_data(data, args.directory)


if __name__ == "__main__":
    main()
