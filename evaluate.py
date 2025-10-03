from sklearn.metrics import classification_report, roc_auc_score, f1_score
import numpy as np

def evaluate_model(model, X_val, y_val, name="Model"):
    print(f"\n{name} Evaluation:")
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    print(classification_report(y_val, y_pred, digits=4))
    print("ROC AUC:", roc_auc_score(y_val, y_proba))

def evaluate_with_threshold(model, X_val, y_val, threshold=0.5, name="Model"):
    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    print(f"\n{name} (threshold={threshold})")
    print(classification_report(y_val, y_pred, digits=4))
    print("ROC AUC:", roc_auc_score(y_val, y_proba))

def find_best_threshold(model, X_val, y_val):
    y_proba = model.predict_proba(X_val)[:, 1]
    best_thresh, best_f1 = 0.5, 0
    for thresh in np.arange(0.1, 0.91, 0.05):
        y_pred = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_val, y_pred)
        if f1 > best_f1:
            best_thresh, best_f1 = thresh, f1
    return best_thresh, best_f1