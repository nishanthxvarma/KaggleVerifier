import os
import re
import pandas as pd
import urllib.request
import zipfile
import tempfile

def is_kaggle_url(url: str) -> bool:
    """Check if the provided string is a valid Kaggle dataset URL."""
    return 'kaggle.com/datasets/' in url

def extract_kaggle_slug(url: str) -> str:
    """Extracts 'username/dataset_name' from Kaggle URL."""
    try:
        # e.g., https://www.kaggle.com/datasets/sudalairajkumar/novel-corona-virus-2019-dataset
        parts = url.strip('/').split('datasets/')[-1].split('/')
        return f"{parts[0]}/{parts[1]}"
    except IndexError:
        return ""

def download_and_read_kaggle_dataset(url: str) -> pd.DataFrame:
    """
    Downloads the first CSV from a Kaggle dataset URL into memory.
    Requires properly configured kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY 
    env variables.
    """
    slug = extract_kaggle_slug(url)
    if not slug:
        raise ValueError("Invalid Kaggle URL format.")

    try:
        import kaggle
        api = kaggle.api
        api.authenticate()
    except (Exception, SystemExit) as e:
        raise Exception(f"Kaggle API Authentication failed. Please ensure 'KAGGLE_USERNAME' and 'KAGGLE_KEY' environment variables are set, or a valid 'kaggle.json' is configured. Original error: {e}")

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Downloading dataset {slug}...")
        api.dataset_download_files(slug, path=temp_dir, unzip=True)
        
        # Find first CSV
        csv_files = []
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.csv'):
                    csv_files.append(os.path.join(root, file))
                    
        if not csv_files:
            raise FileNotFoundError("No CSV files found in the dataset.")
            
        # load first one natively just for demo purposes
        print(f"Reading {csv_files[0]}...")
        # using low_memory=False and robust error handling for weird CSVs
        df = pd.read_csv(csv_files[0], low_memory=False, on_bad_lines='skip')
        return df

def process_upload(file) -> pd.DataFrame:
    """Reads an uploaded CSV file"""
    try:
        df = pd.read_csv(file, low_memory=False, on_bad_lines='skip')
        return df
    except Exception as e:
        raise ValueError(f"Failed to read CSV file: {e}")
