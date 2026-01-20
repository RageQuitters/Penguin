import pandas as pd

# -----------------------------
# 1. Load CSV into DataFrame
# -----------------------------
def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df

# -----------------------------------------
# 2. Drop NA values from DataFrame
# -----------------------------------------
def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with NA values in any column
    df_clean = df.dropna().reset_index(drop=True)


    return df_clean