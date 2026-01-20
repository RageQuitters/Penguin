# FILE TO TEST METHODS DURING DEVELOPMENT

from tools.preprocess import preprocess_dataset, load_dataset
from tools.urgency_model import train_urgency_model
from tools.fault_model import train_fault_model
import joblib
import pandas as pd

# Fake dataset with 5 machines
fake_data = pd.DataFrame({
    "Air temperature [K]": [290.0, 300.5, 302.0, 299.2, 301.1],
    "Process temperature [K]": [30.0, 310.1, 312.5, 307.8, 309.3],
    "Rotational speed [rpm]": [2800, 1600, 1400, 1550, 1490],
    "Torque [Nm]": [4.6, 50.5, 45.0, 48.2, 47.0],
    "Tool wear [min]": [0, 5, 10, 3, 7]
})

df = load_dataset("dataset/train_test.csv")
df_clean = preprocess_dataset(df)
# train_urgency_model(df_clean, [c for c in df.columns if c not in ["UDI","Product ID", "Type", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]])
# train_fault_model(df_clean)
iso = joblib.load("joblib_files/urgency_model.joblib")
urg = joblib.load("joblib_files/fault_model.joblib")
df_clean["anomaly_score"] = -iso.score_samples(df_clean[[c for c in df.columns if c not in ["UDI","Product ID", "Type", "Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]]])
print(df_clean[df_clean["Machine failure"] != 0])
print(df_clean[df_clean["Machine failure"] == 0])

y_pred = urg.predict(fake_data)
# Convert to DataFrame
predicted_faults = pd.DataFrame(y_pred, columns=["TWF","HDF","PWF","OSF","RNF"])

# Add predicted faults to fake data for inspection
fake_data[["pred_TWF","pred_HDF","pred_PWF","pred_OSF","pred_RNF"]] = predicted_faults

pd.set_option('display.max_columns', None)
print(fake_data)