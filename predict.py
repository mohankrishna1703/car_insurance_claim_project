import pandas as pd
import joblib
from preprocess import preprocess_data
from load_data import load_data
import os

def make_predictions():
    # Load trained model
    model_path = "models/best_model.pkl"
    model = joblib.load(model_path)
    print("Best model loaded successfully.")

    # from CSV (using existing test.csv)
    df_test = pd.read_csv("data/test.csv")
    print(f"Loaded test data: {len(df_test)} rows")

    # Preprocess the test data
    df_processed = preprocess_data(df_test)
    print("Preprocessing complete.")

    # Separate features
    X_test = df_processed.drop("is_claim", axis=1, errors='ignore')

    # Make predictions
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Combine results
    results = df_test.copy()
    results["Predicted_is_claim"] = preds
    if probs is not None:
        results["Claim_Probability"] = probs

    # Save predictions to CSV
    os.makedirs("outputs", exist_ok=True)
    results.to_csv("outputs/predictions.csv", index=False)
    print("Predictions saved as 'outputs/predictions.csv'")

    # Show sample predictions
    print("\nSample Predictions:")
    print(results[["Predicted_is_claim", "Claim_Probability"]].head())

if __name__ == "__main__":
    make_predictions()