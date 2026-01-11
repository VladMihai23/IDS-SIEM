import pandas as pd
import joblib
import os



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'processed_data.parquet')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
COLUMN_FILE = os.path.join(MODEL_DIR, 'feature_columns.pkl')


def export():
    print(f"🔍 Looking for dataset at: {DATA_PATH}")

    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: File not found at {DATA_PATH}")
        print("💡 Check if the file name is correct and it is inside the 'dataset' folder.")
        return


    df = pd.read_parquet(DATA_PATH).head(1)


    features = df.drop(['Label', 'Timestamp'], axis=1, errors='ignore')

    column_list = features.columns.tolist()

    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    joblib.dump(column_list, COLUMN_FILE)

    print(f"✅ Success! Saved {len(column_list)} feature names to {COLUMN_FILE}")


if __name__ == "__main__":
    export()