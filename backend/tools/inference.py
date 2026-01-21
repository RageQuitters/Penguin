import os
import joblib
import pandas as pd
from constants import *

def safe_load_model(path: str, model_name: str):
    if not os.path.exists(path):
        print(f"[WARN] {model_name} not found at {path}")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        print(f"[ERROR] Failed to load {model_name}: {e}")
        return None
    
urgency_model = safe_load_model(URGENCY_MODEL_PATH, "Urgency model")
fault_model = safe_load_model(FAULT_MODEL_PATH, "Fault model")
print("model loaded")

def predict_new_point(sensor_tuple: tuple) -> dict:
    """
    sensor_tuple:
    (
        Air temperature [K],
        Process temperature [K],
        Rotational speed [rpm],
        Torque [Nm],
        Tool wear [min]
    )

    Returns:
    {
        "anomaly_score": float | None,
        "fault_types": list[str] | None
    }
    """

    if len(sensor_tuple) != 5:
        raise ValueError("sensor_tuple must contain exactly 5 values")

    df = pd.DataFrame([sensor_tuple], columns=FEATURE_LABELS)

    # Default result
    result = {
        "sensor_data": sensor_tuple,
        "anomaly_score": None,
        "fault_types": None,
    }

    # -----------------------------
    # Anomaly score
    # -----------------------------
    if urgency_model is not None:
        result["anomaly_score"] = -urgency_model.score_samples(df)[0]

    # -----------------------------
    # Fault prediction
    # -----------------------------
    if fault_model is not None:
        pred = fault_model.predict(df)[0]
        result["fault_types"] = [
            fault for fault, val in zip(FAULT_LABELS, pred) if val == 1
        ]

    return result
