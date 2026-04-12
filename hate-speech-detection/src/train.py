"""
train.py
--------
Training pipeline for hate speech detection model.
Handles class imbalance with SMOTE + class_weight.
Compares Logistic Regression, Random Forest, and SVM.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from preprocess import load_and_preprocess, build_tfidf

DATA_PATH = "data/raw/hate_speech.csv"
MODEL_PATH = "models/model.pkl"
TFIDF_PATH = "models/tfidf_vectorizer.pkl"
RANDOM_STATE = 42


def train():
    # ── 1. Load & preprocess ──────────────────────────────────────────────────
    df = load_and_preprocess(DATA_PATH)

    X_raw = df["clean_text"]
    y = df["label"].values

    # ── 2. Train/test split ───────────────────────────────────────────────────
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # ── 3. TF-IDF feature extraction ─────────────────────────────────────────
    X_train, tfidf = build_tfidf(X_train_raw, save_path=TFIDF_PATH)
    X_test = tfidf.transform(X_test_raw)

    # ── 4. Handle class imbalance with SMOTE ─────────────────────────────────
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE — train size: {X_train_res.shape[0]}")

    # ── 5. Model comparison via cross-validation ─────────────────────────────
    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", C=1.0, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "LinearSVC": LinearSVC(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        ),
    }

    print("\n── Cross-Validation Results (5-fold, F1) ──")
    best_score = 0
    best_model = None
    best_name = ""

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for name, model in candidates.items():
        scores = cross_val_score(model, X_train_res, y_train_res, cv=cv, scoring="f1")
        mean_f1 = scores.mean()
        print(f"  {name:<25} F1 = {mean_f1:.4f} ± {scores.std():.4f}")
        if mean_f1 > best_score:
            best_score = mean_f1
            best_model = model
            best_name = name

    print(f"\n✅ Best model: {best_name} (F1 = {best_score:.4f})")

    # ── 6. Final training on full resampled train set ─────────────────────────
    best_model.fit(X_train_res, y_train_res)

    # ── 7. Save model + test split ────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump((X_test, y_test), "models/test_data.pkl")
    print(f"\n💾 Model saved → {MODEL_PATH}")
    print("Run `python src/evaluate.py` for full metrics.")


if __name__ == "__main__":
    train()
