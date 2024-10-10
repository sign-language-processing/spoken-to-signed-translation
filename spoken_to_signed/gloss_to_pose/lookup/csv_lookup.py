import csv
import os

from .lookup import PoseLookup


class CSVPoseLookup(PoseLookup):
    def __init__(self, directory: str, backup: PoseLookup = None):
        if not os.path.exists(directory):
            raise ValueError(f"Directory {directory} does not exist")

        with open(os.path.join(directory, 'index.csv'), mode='r', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))

        super().__init__(rows=rows, directory=directory, backup=backup)
