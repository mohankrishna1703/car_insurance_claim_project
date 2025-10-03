import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from preprocess import build_preprocessor
from models import get_logistic_model, get_random_forest, get_xgboost
from utils import evaluate_model, evaluate_with_threshold, save_submission

RANDOM_SEED = 42

def train():
    # Load data
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    X = train.drop("is_claim", axis=1)
    y = train["is_claim"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
    )

    preprocessor = build_preprocessor(X)

    results = []
    best_model, best_thresh = None, 0.5

    # Logistic Regression
    log_pipe = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_SEED, sampling_strategy=0.2)),
        ("clf", get_logistic_model())
    ])
    log_pipe.fit(X_train, y_train)
    evaluate_model(log_pipe, X_val, y_val, "Logistic Regression")
    evaluate_with_threshold(log_pipe, X_val, y_val, 0.3, "Logistic Regression")
    results.append((log_pipe, 0.5))

    # Random Forest
    rf_pipe = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_SEED, sampling_strategy=0.2)),
        ("clf", get_random_forest())
    ])
    rf_pipe.fit(X_train, y_train)
    evaluate_model(rf_pipe, X_val, y_val, "Random Forest")
    evaluate_with_threshold(rf_pipe, X_val, y_val, 0.3, "Random Forest")
    results.append((rf_pipe, 0.5))

    # XGBoost
    xgb_pipe = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=RANDOM_SEED, sampling_strategy=0.2)),
        ("clf", get_xgboost())
    ])
    xgb_pipe.fit(X_train, y_train)
    evaluate_model(xgb_pipe, X_val, y_val, "XGBoost")
    evaluate_with_threshold(xgb_pipe, X_val, y_val, 0.3, "XGBoost")

    # Use default threshold = 0.5
    results.append((xgb_pipe, 0.5))

    # Choose XGBoost as final (best ROC-AUC in tuning)
    best_model, best_thresh = xgb_pipe, 0.5

    # Save submission
    save_submission(best_model, test, threshold=best_thresh, name="submission.csv")

    return best_model, best_thresh