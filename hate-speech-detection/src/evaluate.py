"""
evaluate.py
-----------
Loads saved model and test data, prints full classification metrics,
and plots confusion matrix + ROC curve.
"""

import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score
)
import os

MODEL_PATH = "models/model.pkl"
TEST_DATA_PATH = "models/test_data.pkl"
OUTPUT_DIR = "outputs"


def evaluate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model and test data
    model = joblib.load(MODEL_PATH)
    X_test, y_test = joblib.load(TEST_DATA_PATH)

    y_pred = model.predict(X_test)

    # Decision scores for ROC (LinearSVC uses decision_function)
    if hasattr(model, "predict_proba"):
        y_scores = model.predict_proba(X_test)[:, 1]
    else:
        y_scores = model.decision_function(X_test)

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_scores)

    print("=" * 50)
    print(f"  Accuracy : {acc:.4f}")
    print(f"  AUC-ROC  : {auc:.4f}")
    print("=" * 50)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Not Hate", "Hate Speech"]))

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Not Hate", "Hate Speech"],
                yticklabels=["Not Hate", "Hate Speech"])
    plt.title("Confusion Matrix — Hate Speech Detection")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()
    print(f"✅ Confusion matrix saved → {OUTPUT_DIR}/confusion_matrix.png")

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC Curve (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Hate Speech Detection")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/roc_curve.png", dpi=150)
    plt.close()
    print(f"✅ ROC curve saved → {OUTPUT_DIR}/roc_curve.png")


if __name__ == "__main__":
    evaluate()
