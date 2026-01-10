import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import confusion_matrix, classification_report

# --- Robust Path Management ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'attack_detector.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'label_encoder.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'processed_data.parquet')


def evaluate_and_visualize():
    print("--- Phase 3: Model Evaluation & Visualization ---")

    # Load the Model, Encoder, and a sample of the data
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        print("ERROR: Model or Encoder not found in /models. Run 02_training.py first.")
        return

    print("Loading model and data...")
    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)

    # We use a sample of 100,000 rows for evaluation to keep it fast
    df = pd.read_parquet(DATA_PATH).sample(100000, random_state=42)

    X = df.drop(['Label', 'Timestamp'], axis=1, errors='ignore')
    y_true = le.transform(df['Label'])

    # Generate Predictions
    print("Generating predictions on test sample...")
    y_pred = model.predict(X)

    # Create Confusion Matrix Plot
    print("Creating Confusion Matrix...")
    plt.figure(figsize=(12, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix - Network Attack Detection')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    # Save the plot
    cm_plot_path = os.path.join(BASE_DIR, 'results', 'confusion_matrix.png')
    plt.savefig(cm_plot_path)
    print(f"Saved: {cm_plot_path}")

    # Create Feature Importance Plot
    print("Calculating Feature Importance...")
    plt.figure(figsize=(10, 6))
    importances = pd.Series(model.feature_importances_, index=X.columns)
    importances.nlargest(10).plot(kind='barh', color='darkorange')
    plt.title('Top 10 Most Important Network Features')
    plt.xlabel('Importance Score')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    # Save the plot
    fi_plot_path = os.path.join(BASE_DIR, 'results', 'feature_importance.png')
    plt.savefig(fi_plot_path)
    print(f"Saved: {fi_plot_path}")

    # Print Classification Report to Console
    print("\n--- Final Classification Report ---")
    print(classification_report(y_true, y_pred, target_names=le.classes_))


if __name__ == "__main__":
    evaluate_and_visualize()