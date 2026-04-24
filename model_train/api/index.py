from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
import os
from pydantic import BaseModel
from mangum import Mangum

app = FastAPI()

# ── Load models once at startup ──────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.abspath(os.path.join(BASE, "..", "model_train"))

if not os.path.exists(MODEL_DIR):
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

# ── Routes ──────────────────────────────────────────────────────────────────
@app.post("/predict")
def predict(m: MachineInput):
    X = pd.DataFrame(
        [{
            "Air temperature [K]": m.air_temperature,
            "Process temperature [K]": m.process_temperature,
            "Rotational speed [rpm]": m.rotational_speed,
            "Torque [Nm]": m.torque,
            "Tool wear [min]": m.tool_wear,
        }]
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

# ── Vercel handler (CRITICAL FIX) ───────────────────────────────────────────
asgi_handler = Mangum(app)

def handler(request, context):
    return asgi_handler(request, context)