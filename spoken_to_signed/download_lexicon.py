import argparse
import csv
import os
from datetime import datetime
from typing import List, Dict

from pose_format import PoseHeader, Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.utils.reader import BufferReader
from tqdm import tqdm

LEXICON_INDEX = ['path', 'spoken_language', 'signed_language', 'start', 'end', 'words', 'glosses', 'priority']


def init_index(index_path: str):
    if not os.path.isfile(index_path):
        # Create csv file with specified header
        with open(index_path, 'w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(LEXICON_INDEX)


def load_signsuisse(directory_path: str) -> List[Dict[str, str]]:
    try:
        import sign_language_datasets
    except ImportError as e:
        raise ImportError("Please install sign_language_datasets. pip install sign-language-datasets") from e

    import tensorflow_datasets as tfds
    # noinspection PyUnresolvedReferences
    import sign_language_datasets.datasets.signsuisse as signsuisse
    # noinspection PyUnresolvedReferences
    from sign_language_datasets.datasets.signsuisse.signsuisse import _POSE_HEADERS
    from sign_language_datasets.datasets.config import SignDatasetConfig

    IANA_TAGS = {
        "ch-de": "sgg",
        "ch-fr": "ssr",
        "ch-it": "slf",
    }

    # for cache busting, we use today's date
    date_str = datetime.now().strftime("%Y-%m-%d")
    config = SignDatasetConfig(name=date_str, version="1.0.0", include_video=False, include_pose="holistic")
    dataset = tfds.load(name='sign_suisse', builder_kwargs={"config": config})

    with open(_POSE_HEADERS["holistic"], "rb") as buffer:
        pose_header = PoseHeader.read(BufferReader(buffer.read()))

    for datum in tqdm(dataset["train"]):
        uid_raw = datum['id'].numpy().decode('utf-8')
        spoken_language = datum['spokenLanguage'].numpy().decode('utf-8')
        signed_language = IANA_TAGS[datum['signedLanguage'].numpy().decode('utf-8')]
        words = datum['name'].numpy().decode('utf-8')

        # Load pose and save to file
        tf_pose = datum['pose']
        fps = int(tf_pose["fps"].numpy())
        if fps == 0:
            continue
        pose_body = NumPyPoseBody(fps, tf_pose["data"].numpy(), tf_pose["conf"].numpy())
        pose = Pose(pose_header, pose_body)
        pose_relative_path = os.path.join(signed_language, f"{uid_raw}.pose")
        os.makedirs(os.path.join(directory_path, signed_language), exist_ok=True)
        with open(os.path.join(directory_path, pose_relative_path), "wb") as f:
            pose.write(f)

        yield {
            'path': pose_relative_path,
            'spoken_language': spoken_language,
            'signed_language': signed_language,
            'words': words,
            'start': "0",
            'end': str(len(pose_body.data) / fps),  # pose duration
            'glosses': "",
            'priority': "",
        }


def normalize_row(row: Dict[str, str]):
    if row['glosses'] == "" and row['words'] != "":
        from spoken_to_signed.text_to_gloss.simple import text_to_gloss
        try:
            sentences = text_to_gloss(text=row['words'], language=row['spoken_language'])
            glosses = [g for sentence in sentences for w, g in sentence]
            row['glosses'] = " ".join(glosses)
        except ValueError as e:
            if not ('Language' in str(e) and 'not supported' in str(e)):
                raise e


def get_data(name: str, directory: str):
    data_loaders = {
        'signsuisse': load_signsuisse,
    }
    if name not in data_loaders:
        raise NotImplementedError(f"{name} is unknown.")

    return data_loaders[name](directory)


def add_data(data: List[Dict[str, str]], directory: str):
    index_path = os.path.join(directory, 'index.csv')
    os.makedirs(directory, exist_ok=True)
    init_index(index_path)

    with open(index_path, 'a', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        for row in tqdm(data):
            normalize_row(row)
            writer.writerow([row[key] for key in LEXICON_INDEX])

    print(f"Added entries to {index_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", choices=['signsuisse'], required=True)
    parser.add_argument("--directory", type=str, required=True)
    args = parser.parse_args()

    data = get_data(args.name, args.directory)
    add_data(data, args.directory)


if __name__ == '__main__':
    main()
