import csv
from pathlib import Path

from google.cloud import storage

# Initialize Google Cloud Storage client
client = storage.Client()

# Define bucket name
bucket_name = 'sign-mt-poses'
bucket = client.bucket(bucket_name)


def download_file_from_gcs(bucket, source_filename, destination_path):
    """Download a file from GCS."""
    blob = bucket.blob(source_filename)
    blob.download_to_filename(destination_path)
    print(f"Downloaded {source_filename} to {destination_path}")


# Open the CSV file and process each row
csv_file_path = Path('data.csv')

with csv_file_path.open('r') as csvfile:
    reader = csv.DictReader(csvfile)

    for row in reader:
        # Extract the path and filename
        full_path = Path(row['path'])
        if full_path.exists():
            continue

        filename = full_path.name
        Path(row['signed_language']).mkdir(exist_ok=True)

        # Download the file from GCS to the specified path
        download_file_from_gcs(bucket, filename, str(full_path))
