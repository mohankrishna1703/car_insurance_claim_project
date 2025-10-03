# predict.py
import pandas as pd
import joblib

def predict_and_save(model, threshold=0.5):
    test_df = pd.read_csv("data/test.csv")
    y_proba = model.predict_proba(test_df)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    sub = pd.DataFrame({
        "policy_id": test_df["policy_id"],
        "is_claim": y_pred
    })
    sub.to_csv("submission.csv", index=False)
    print("Submission saved to: submission.csv")