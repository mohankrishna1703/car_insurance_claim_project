import pandas as pd
import joblib
from sklearn.metrics import classification_report, roc_auc_score

def evaluate_model(model, X_val, y_val, name):
    print(f"\n{name} Evaluation:")
    preds = model.predict(X_val)
    probs = model.predict_proba(X_val)[:, 1]
    print(classification_report(y_val, preds))
    print("ROC AUC:", roc_auc_score(y_val, probs))

def evaluate_with_threshold(model, X_val, y_val, threshold, name):
    print(f"\n{name} (threshold={threshold})")
    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs >= threshold).astype(int)
    print(classification_report(y_val, preds))
    print("ROC AUC:", roc_auc_score(y_val, probs))

def save_submission(model, test_df, threshold=0.5, name="submission.csv"):
    probs = model.predict_proba(test_df)[:, 1]
    preds = (probs >= threshold).astype(int)
    submission = pd.DataFrame({
        "policy_id": test_df["policy_id"],
        "is_claim": preds
    })
    submission.to_csv(name, index=False)
    joblib.dump(model, "final_model.joblib")
    print(f"Submission saved to: {name}")