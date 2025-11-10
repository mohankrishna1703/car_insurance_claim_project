import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from load_data import load_data
from preprocess import preprocess_data

def evaluate_model():
    # Load training data
    df = load_data("train_data")
    print(f"Loaded {len(df)} rows from train_data")

    # Preprocess data
    df = preprocess_data(df)
    print("Preprocessing done\n")

    # Split into features and target
    X = df.drop("is_claim", axis=1)
    y = df["is_claim"]

    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load best model
    model = joblib.load("models/best_model.pkl")
    print("Loaded best model for evaluation.\n")

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred)

    print("Model Evaluation Results:")
    print(f"Accuracy : {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall   : {recall:.3f}")
    print(f"F1 Score : {f1:.3f}")
    print(f"ROC-AUC  : {roc_auc:.3f}\n")

    # Classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Evaluation completed successfully.\n")

if __name__ == "__main__":
    evaluate_model()