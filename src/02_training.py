import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

#Setting the path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(BASE_DIR, 'dataset', 'processed_data.parquet')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'attack_detector.pkl')
ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.pkl')

def train_ai():
    print("--- Phase 2: Training the AI Model ---")
    df = pd.read_parquet(INPUT_FILE)

    # Only a sample of 300k
    df_sample = df.sample(n=300000, random_state=42)

    X = df_sample.drop(['Label', 'Timestamp'], axis=1, errors='ignore')
    y = df_sample['Label']

    # Turning the text in numbers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    print("Training Random Forest... (taking around 2 minutes to do the RF)")
    model = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)

    # Saving the result
    if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)

    print(f"SUCCESS! Accuracy: {model.score(X_test, y_test):.4f}")
    print(f"Model saved in: {MODEL_DIR}")

if __name__ == "__main__":
    train_ai()