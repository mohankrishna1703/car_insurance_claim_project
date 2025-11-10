import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from load_data import load_data
from preprocess import preprocess_data

def train_models():
    # Load data
    df = load_data("train_data")
    df = preprocess_data(df)

    # Split into features and target
    X = df.drop(columns=["is_claim"])
    y = df["is_claim"].astype(int)

    # Split train-test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Handle class imbalance
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: Class balance -> {y_train.value_counts().to_dict()}")

    # Define models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=10000),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    # Train and evaluate models
    scores = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        combined = (acc + auc) / 2
        scores[name] = combined

        print(f"   Accuracy: {acc:.3f} | ROC-AUC: {auc:.3f} | Combined: {combined:.3f}")

        joblib.dump(model, f"models/{name}.pkl")

    # Select best model
    best_model = max(scores, key=scores.get)
    joblib.dump(models[best_model], "models/best_model.pkl")

    print(f"\nBest Model: {best_model} (Based on Accuracy + ROC-AUC)")

if __name__ == "__main__":
    train_models()