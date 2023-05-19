import argparse
import csv
import os

LEXICON_INDEX = ['path', 'spoken_language', 'signed_language', 'words', 'glosses', 'priority']


def init_index(index_path: str):
    if not os.path.isfile(index_path):
        # Create csv file with specified header
        with open(index_path, 'w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(LEXICON_INDEX)


def main(name: str, directory: str):
    index_path = os.path.join(directory, 'index.csv')
    init_index(index_path)

    raise NotImplementedError(f"{name} is unknown.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", choices=['signsuisse'], required=True)
    parser.add_argument("--directory", type=str, required=True)
    args = parser.parse_args()

    main(args.name, args.directory)
