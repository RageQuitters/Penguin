import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss, f1_score
import joblib

from constants import *

def train_fault_model(df, test_size=0.2, random_state=42, n_estimators=200):
    """
    Train a multi-label Random Forest model to predict machine faults.
    Saves the trained model to 'joblib_files/fault_model.joblib'.

    Parameters:
    - df: pandas DataFrame, the dataset
    - test_size: float, fraction of data for test split
    - random_state: int, for reproducibility
    - n_estimators: int, number of trees in Random Forest

    Returns:
    - trained MultiOutputClassifier
    - X_test, y_test for evaluation
    - y_pred on X_test
    """

    # -----------------------------
    # 1. Ensure model directory exists
    # -----------------------------
    model_dir = "joblib_files"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "fault_model.joblib")

    # -----------------------------
    # 2. Features and target
    # -----------------------------
    X = df[FEATURE_LABELS]
    y = df[FAULT_LABELS]

    # -----------------------------
    # 3. Train-test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # -----------------------------
    # 4. Train model
    # -----------------------------
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    multi_rf = MultiOutputClassifier(rf)
    multi_rf.fit(X_train, y_train)

    # -----------------------------
    # 5. Predict on test set
    # -----------------------------
    y_pred = multi_rf.predict(X_test)

    # -----------------------------
    # 6. Evaluation
    # -----------------------------
    print("Hamming Loss:", hamming_loss(y_test, y_pred))
    print("F1 Score (micro):", f1_score(y_test, y_pred, average='micro'))
    print("F1 Score (macro):", f1_score(y_test, y_pred, average='macro'))

    # -----------------------------
    # 7. Save model
    # -----------------------------
    joblib.dump(multi_rf, model_path)
    print(f"Trained model saved to {model_path}")

    return multi_rf, X_test, y_test, y_pred
