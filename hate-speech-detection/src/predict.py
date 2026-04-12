"""
predict.py
----------
Inference script — classify any text as hate speech or not.
Usage:
    python src/predict.py --text "your text here"
"""

import argparse
import joblib
from preprocess import clean_text

MODEL_PATH = "models/model.pkl"
TFIDF_PATH = "models/tfidf_vectorizer.pkl"

LABELS = {0: "✅ Not Hate Speech", 1: "🚫 Hate Speech Detected"}


def predict(text: str) -> dict:
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)

    cleaned = clean_text(text)
    features = tfidf.transform([cleaned])
    pred = model.predict(features)[0]

    result = {
        "input": text,
        "cleaned": cleaned,
        "prediction": pred,
        "label": LABELS[pred],
    }

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(features)[0]
        result["confidence"] = f"{max(prob) * 100:.1f}%"
    elif hasattr(model, "decision_function"):
        score = model.decision_function(features)[0]
        result["decision_score"] = f"{score:.4f}"

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hate Speech Classifier")
    parser.add_argument("--text", type=str, required=True, help="Text to classify")
    args = parser.parse_args()

    result = predict(args.text)
    print("\n── Prediction Result ──────────────────────")
    for k, v in result.items():
        print(f"  {k:<18}: {v}")
    print("───────────────────────────────────────────\n")
