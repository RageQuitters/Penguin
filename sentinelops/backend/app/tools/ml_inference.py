"""
ML Inference Tool
-----------------
Wraps the real trained Isolation Forest (urgency) and Random Forest (fault)
joblib models. This is the bridge between raw sensor data and ML scores.

In production, swap joblib.load() for ModelArts inference endpoint calls.
"""
import os
import joblib
import pandas as pd
from functools import lru_cache
from app.core.config import get_settings

settings = get_settings()

FEATURE_LABELS = settings.feature_labels
FAULT_LABELS = settings.fault_labels


@lru_cache(maxsize=1)
def _load_urgency_model():
    path = settings.urgency_model_path
    if not os.path.exists(path):
        raise FileNotFoundError(f"Urgency model not found at {path}")
    return joblib.load(path)


@lru_cache(maxsize=1)
def _load_fault_model():
    path = settings.fault_model_path
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fault model not found at {path}")
    return joblib.load(path)


def run_isolation_forest(reading: tuple) -> float:
    """
    Score a sensor reading against the trained Isolation Forest.
    Returns anomaly score in [0, 1] — higher = more anomalous.

    Production swap: POST to ModelArts /urgency endpoint.
    """
    model = _load_urgency_model()
    df = pd.DataFrame([reading], columns=FEATURE_LABELS)
    # sklearn returns negative scores; negate so higher = more anomalous
    raw = model.score_samples(df)[0]
    score = float(-raw)
    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


def run_fault_classifier(reading: tuple) -> list[str]:
    """
    Predict active fault types for a sensor reading.
    Returns a list of fault label strings, e.g. ['TWF', 'PWF'].

    Production swap: POST to ModelArts /fault endpoint.
    """
    model = _load_fault_model()
    df = pd.DataFrame([reading], columns=FEATURE_LABELS)
    pred = model.predict(df)[0]
    return [fault for fault, val in zip(FAULT_LABELS, pred) if val == 1]
