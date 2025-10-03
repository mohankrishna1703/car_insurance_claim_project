import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from preprocess import build_preprocessor
from xgboost import XGBClassifier

def tune_xgb():
    print("🔎 Running hyperparameter tuning (this may take some time)...")

    train = pd.read_csv("data/train.csv")
    X = train.drop(columns=["is_claim"])
    y = train["is_claim"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    preprocessor = build_preprocessor(X_train)
    smote = SMOTE(sampling_strategy=0.3, random_state=42)

    xgb = XGBClassifier(eval_metric="logloss", use_label_encoder=False, random_state=42)

    pipe = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", smote),
        ("clf", xgb)
    ])

    param_grid = {
        "clf__n_estimators": [200, 400],
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [3, 6],
        "clf__subsample": [0.6, 0.8],
        "clf__colsample_bytree": [0.6, 0.8],
        "clf__scale_pos_weight": [12, 16],
    }

    grid = GridSearchCV(pipe, param_grid, cv=3, scoring="roc_auc", verbose=2, n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Best ROC-AUC score from CV:", grid.best_score_)
    print("Best parameters:", grid.best_params_)

if __name__ == "__main__":
    tune_xgb()