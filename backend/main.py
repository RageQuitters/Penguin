# FILE TO TEST METHODS DURING DEVELOPMENT

from tools.preprocess import preprocess_dataset, load_dataset
from tools.urgency_model import train_urgency_model
from tools.fault_model import train_fault_model
from tools.inference import predict_new_point, safe_load_model
from constants import *
import joblib
import pandas as pd

df = load_dataset("dataset/train_test.csv")
df_clean = preprocess_dataset(df)

# Comment out after training models successfully once.
train_fault_model(df_clean)
train_urgency_model(df_clean)

print(predict_new_point((290.0, 30.0, 2500, 4, 5)))
print(predict_new_point((290.0, 30.0, 2500, 4, 6)))
