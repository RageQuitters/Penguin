from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
import uvicorn
import os
from pydantic import BaseModel

app = FastAPI()

# ── Load models once at startup ──────────────────────────────────────────────
# Vercel will import this file and look for the FastAPI instance named `app`.
# Locally, this still works with: python your_file.py

BASE = os.path.dirname(os.path.abspath(__file__))

# Works if this file is in something like:
#   api/ml_server.py
# and model_train is at repo root:
#   model_train/scaler.joblib
#
# If your structure is different, adjust this path.
MODEL_DIR = os.path.abspath(os.path.join(BASE, "..", "model_train"))

if not os.path.exists(MODEL_DIR):
    # Fallback for your original local structure:
    # server/api/ml_server.py -> ../../model_train
    MODEL_DIR = os.path.abspath(os.path.join(BASE, "../../model_train"))

scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
lof = joblib.load(os.path.join(MODEL_DIR, "lof_model.joblib"))
rf_models = joblib.load(os.path.join(MODEL_DIR, "rf_models.joblib"))
feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.joblib"))

print("Models loaded:", list(rf_models.keys()))


# ── Helpers ─────────────────────────────────────────────────────────────────
def get_anomaly_score(model, X):
    raw = model.decision_function(X)
    return 1 / (1 + np.exp(-raw))


def final_decision(anomaly_score: float, failure_vector: list) -> str:
    if anomaly_score > 0.7:
        return "FAILURE"
    elif anomaly_score > 0.4 or sum(failure_vector) > 0:
        return "WARNING"
    else:
        return "NORMAL"


# ── Request schema ──────────────────────────────────────────────────────────
class MachineInput(BaseModel):
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float


# ── /predict ────────────────────────────────────────────────────────────────
@app.post("/predict")
def predict(m: MachineInput):
    X = pd.DataFrame(
        [
            {
                "Air temperature [K]": m.air_temperature,
                "Process temperature [K]": m.process_temperature,
                "Rotational speed [rpm]": m.rotational_speed,
                "Torque [Nm]": m.torque,
                "Tool wear [min]": m.tool_wear,
            }
        ]
    )[feature_cols]

    X_scaled = scaler.transform(X)

    anomaly_score = float(get_anomaly_score(lof, X_scaled)[0])

    targets = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    failure_vector = [int(rf_models[t].predict(X_scaled)[0]) for t in targets]

    fault_dict = dict(zip(targets, failure_vector))
    active_faults = [t for t, v in fault_dict.items() if v == 1]

    decision = final_decision(anomaly_score, failure_vector)

    return {
        "anomaly_score": anomaly_score,
        "failure_vector": fault_dict,
        "active_faults": active_faults,
        "decision": decision,
    }


@app.get("/health")
def health():
    return {"ok": True}


# ── Local-only runner ───────────────────────────────────────────────────────
# Vercel imports `app` directly.
# Do not run uvicorn on Vercel.
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    uvicorn.run(app, host="0.0.0.0", port=port)