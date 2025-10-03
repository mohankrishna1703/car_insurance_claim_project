from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def get_logistic_model():
    return LogisticRegression(max_iter=1000, class_weight="balanced")

def get_random_forest():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42
    )

def get_xgboost():
    """Return tuned XGBoost classifier with best params from tune_xgb.py"""
    return XGBClassifier(
        n_estimators=400,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.6,
        colsample_bytree=0.8,
        scale_pos_weight=16,
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False
    )