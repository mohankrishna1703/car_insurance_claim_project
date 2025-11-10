import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df):
    df = df.copy()

    # Fill missing values
    df.fillna(0, inplace=True)

    # Convert all categorical columns using LabelEncoder
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Scale numeric columns (except target)
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    if "is_claim" in numeric_cols:
        numeric_cols = numeric_cols.drop("is_claim")

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    print("Preprocessing done")
    return df