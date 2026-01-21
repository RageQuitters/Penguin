from sklearn.ensemble import IsolationForest
import joblib
import os
from constants import *

# Train anomaly detector on normal historical data (rows without machine failure)
def train_urgency_model(df):
    model_dir = "joblib_files"
    model_path = os.path.join(model_dir, "urgency_model.joblib")
    
    # Create folder if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    normal_data = df[df["Machine failure"] == 0][FEATURE_LABELS]
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(normal_data)
    print(f"Isolation Forest saved to {model_path}")
    joblib.dump(iso, model_path)
    return iso


