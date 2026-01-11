import psycopg2
import pandas as pd

# Database configuration
DB_CONFIG = {
    "dbname": "siem_db",
    "user": "postgres",
    "password": "Vlad5432",
    "host": "localhost"
}

def get_connection():
    return psycopg2.connect(**DB_CONFIG)

def create_tables():
    """Create the table structure"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS siem_alerts (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            attack_type VARCHAR(100),
            confidence FLOAT,
            status VARCHAR(50)
        );
    """)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Database tables initialized successfully.")

def insert_alert(attack_type, confidence):
    """Call this from the Dashboard or Sniffer to save an alert."""
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO siem_alerts (attack_type, confidence, status) VALUES (%s, %s, %s)",
            (attack_type, confidence, "BLOCKED")
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ DB Insert Error: {e}")

def get_all_logs():
    """Fetch logs to display in the Streamlit UI."""
    conn = get_connection()
    query = "SELECT timestamp, attack_type, confidence, status FROM siem_alerts ORDER BY timestamp DESC LIMIT 100"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


if __name__ == "__main__":
    create_tables()