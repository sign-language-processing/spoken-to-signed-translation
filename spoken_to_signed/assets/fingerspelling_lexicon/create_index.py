# load data.csv and save the same as index.csv

import csv

from pathlib import Path

with Path('data.csv').open('r') as data_file:
    rows = list(csv.DictReader(data_file))

# Here we modify rows to, for example, mirror some of the data for other languages
for row in rows:
    if row['spoken_language'] == 'en':
        # Duplicate all ASL for french and swiss-french
        for spoken_language, signed_language in [('fr', 'fsl')]:
            new_row = row.copy()
            new_row['spoken_language'] = spoken_language
            new_row['signed_language'] = signed_language
            rows.append(new_row)

with Path('index.csv').open('w', newline='') as index_file:
    writer = csv.DictWriter(index_file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)