import pandas as pd
import numpy as np
import os

# Line to search where the .py file is exactly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_DIR = os.path.join(BASE_DIR, 'dataset')
OUTPUT_FILE = os.path.join(INPUT_DIR, 'processed_data.parquet')

print(f"Working Directory: {BASE_DIR}")
print(f"Looking for data in: {INPUT_DIR}")


def preprocess_network_data():
    print("--- Phase 1: Data Ingestion & Cleaning ---")

    files = ['Bruteforce.parquet',
             'DDoS.parquet',
             'Web.parquet']

    data_list = []
    for file in files:
        path = os.path.join(INPUT_DIR, file)
        if os.path.exists(path):
            print(f"Reading: {file}...")
            # Loading the data
            df_temp = pd.read_parquet(path)
            print(f"   -> Successfully loaded {len(df_temp)} rows.")
            data_list.append(df_temp)
        else:
            print(f"!!! File not found: {file} in {INPUT_DIR}")

    if not data_list:
        print("!!! ERROR: No data found in the dataset directory.")
        return

    # Merge all the data detections to one big table
    df = pd.concat(data_list, ignore_index=True)
    print(f"Total rows before cleaning: {len(df)}")

    # Standardize column names (Corrected logic)
    df.columns = df.columns.str.strip()

    # Fix technical errors: Inf / NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    print(f"Total rows after cleaning: {len(df)}")

    # Showing the attack distribution
    if 'Label' in df.columns:
        print("\nAttack classes detected:")
        print(df['Label'].value_counts())

    # Save for the AI Training Phase
    print(f"Saving processed data to: {OUTPUT_FILE}...")
    df.to_parquet(OUTPUT_FILE)
    print(f"\nSUCCESS: Cleaned dataset saved to {OUTPUT_FILE}")



if __name__ == "__main__":
    preprocess_network_data()