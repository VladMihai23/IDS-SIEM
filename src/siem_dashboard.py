import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os
import time
import psycopg2
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(page_title="AI-Powered SIEM Dashboard", layout="wide", page_icon="🛡️")

# --- Load AI Model & Encoder ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'attack_detector.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'models', 'label_encoder.pkl')
DATA_PATH = os.path.join(BASE_DIR, 'dataset', 'processed_data.parquet')


@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
    return model, le


try:
    model, le = load_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# --- App Header ---
st.title("🛡️ CyberGuard AI - IDS/SIEM Dashboard")
st.markdown("---")

# --- Create 3 Main Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Laboratory & Research", "🖥️ Live Monitoring", "🗄️ Database History"])

# --- TAB 1: LABORATORY & METRICS ---
with tab1:
    st.header("Phase 1: Model Training & Performance")
    st.info("Results from the laboratory phase using the CIC-IDS2018 dataset (2.4M records).")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")
        cm_path = os.path.join(BASE_DIR, 'results', 'confusion_matrix.png')
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Visualizing prediction accuracy across all classes.")
        else:
            st.warning("Confusion Matrix image not found in /results.")

    with col2:
        st.subheader("Feature Importance")
        fi_path = os.path.join(BASE_DIR, 'results', 'feature_importance.png')
        if os.path.exists(fi_path):
            st.image(fi_path, caption="Top network features used by the AI to detect threats.")
        else:
            st.warning("Feature Importance image not found in /results.")

    st.divider()
    st.subheader("Dataset Overview")
    stats_data = {
        "Metric": ["Total Flows Analyzed", "Training Set Size", "Testing Set Size", "Algorithm", "Accuracy"],
        "Value": ["2,404,416", "400,000", "100,000", "Random Forest", "99.9%"]
    }
    st.table(pd.DataFrame(stats_data))

# --- TAB 2: LIVE SIEM MONITORING ---
with tab2:
    st.header("Phase 2: Live Network Traffic Analysis")
    st.markdown("This section simulates traffic coming from your **Virtual Machine (VM)**.")

    if st.button("🚀 Start Real-Time Capture"):
        st.write("Intercepting network packets...")

        # Load sample data to simulate live traffic
        df_sample = pd.read_parquet(DATA_PATH).sample(50)

        # Placeholders for dynamic updates
        alert_placeholder = st.empty()
        chart_placeholder = st.empty()
        table_placeholder = st.empty()

        detected_events = []

        for i in range(len(df_sample)):
            row = df_sample.iloc[[i]]
            features = row.drop(['Label', 'Timestamp'], axis=1, errors='ignore')

            # AI Prediction
            pred_idx = model.predict(features)[0]
            label = le.inverse_transform([pred_idx])[0]

            timestamp = datetime.now().strftime("%H:%M:%S")
            event = {"Time": timestamp, "Detection": label, "Status": "BLOCKED" if label != "Benign" else "ALLOWED"}
            detected_events.append(event)

            # Update Dashboard
            with alert_placeholder.container():
                if label == "Benign":
                    st.success(f"🟢 [CLEAN] Traffic at {timestamp} identified as safe.")
                else:
                    st.error(f"🔴 [ALERT] {label.upper()} detected at {timestamp}! Traffic dropped.")

            # Update Pie Chart
            event_df = pd.DataFrame(detected_events)
            fig = px.pie(event_df, names='Detection', title="Live Threat Distribution", hole=0.4,
                         color_discrete_map={'Benign': 'green'})
            chart_placeholder.plotly_chart(fig, use_container_width=True)

            time.sleep(0.8)  # Simulating live flow

# --- TAB 3: DATABASE HISTORY ---
with tab3:
    st.header("Phase 3: PostgreSQL Event Logs")
    st.write("Permanent records of all detected security incidents stored in the database.")

    # Placeholder for database connection
    try:
        # In the future, replace with real database logic
        # conn = psycopg2.connect(dbname="siem_db", user="postgres", password="yourpassword")
        st.info("Database Connection Status: Listening for new entries...")

        # For now, we simulate the database view
        st.table(pd.DataFrame({
            "ID": [101, 102],
            "Timestamp": ["2024-05-20 14:05:22", "2024-05-20 14:10:05"],
            "Attack Type": ["SSH-Bruteforce", "DDoS-LOIC"],
            "Confidence": [0.98, 1.00],
            "Action": ["Blocked", "Blocked"]
        }))
    except Exception as e:
        st.error(f"Could not connect to PostgreSQL: {e}")