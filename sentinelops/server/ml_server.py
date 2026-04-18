from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
import uvicorn
import os
from pydantic import BaseModel

app = FastAPI()

# ── Load models once at startup ──────────────────────────────────────────────
# Resolves to Penguin/model_train/ regardless of where you run this script from
BASE      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, "../../model_train")

scaler       = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
lof          = joblib.load(os.path.join(MODEL_DIR, "lof_model.joblib"))
rf_models    = joblib.load(os.path.join(MODEL_DIR, "rf_models.joblib"))   # dict: TWF/HDF/PWF/OSF/RNF
feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.joblib"))

print("✅ Models loaded:", list(rf_models.keys()))

# ── Helpers (copied directly from your notebook) ─────────────────────────────
def get_anomaly_score(model, X):
    raw = model.decision_function(X)
    return 1 / (1 + np.exp(-raw))   # sigmoid normalisation

def final_decision(anomaly_score: float, failure_vector: list) -> str:
    if anomaly_score > 0.7:
        return "FAILURE"
    elif anomaly_score > 0.4 or sum(failure_vector) > 0:
        return "WARNING"
    else:
        return "NORMAL"

# ── Request schema ────────────────────────────────────────────────────────────
class MachineInput(BaseModel):
    air_temperature: float       # Air temperature [K]
    process_temperature: float   # Process temperature [K]
    rotational_speed: float      # Rotational speed [rpm]
    torque: float                # Torque [Nm]
    tool_wear: float             # Tool wear [min]

# ── /predict  ─────────────────────────────────────────────────────────────────
@app.post("/predict")
def predict(m: MachineInput):
    # Build DataFrame with the exact column names the scaler expects
    X = pd.DataFrame([{
        "Air temperature [K]":     m.air_temperature,
        "Process temperature [K]": m.process_temperature,
        "Rotational speed [rpm]":  m.rotational_speed,
        "Torque [Nm]":             m.torque,
        "Tool wear [min]":         m.tool_wear,
    }])[feature_cols]   # reorder to match training order

    X_scaled = scaler.transform(X)

    # ── Stage 1: LOF anomaly score ────────────────────────────────────────
    anomaly_score = float(get_anomaly_score(lof, X_scaled)[0])

    # ── Stage 2: RF fault classifiers ────────────────────────────────────
    targets = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    failure_vector = [int(rf_models[t].predict(X_scaled)[0]) for t in targets]
    fault_dict     = dict(zip(targets, failure_vector))
    active_faults  = [t for t, v in fault_dict.items() if v == 1]

    # ── Stage 3: final decision (your notebook logic) ─────────────────────
    decision = final_decision(anomaly_score, failure_vector)

    return {
        "anomaly_score":  anomaly_score,
        "failure_vector": fault_dict,        # { TWF: 0, HDF: 1, ... }
        "active_faults":  active_faults,     # ["HDF"]
        "decision":       decision,          # "NORMAL" | "WARNING" | "FAILURE"
    }

@app.get("/health")
def health():
    return {"ok": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5001)