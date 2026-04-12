# 🚫 Hate Speech Detection

> NLP classification model achieving **92% accuracy** in detecting hate speech from text data.  
> Built during Data Science Industrial Program at **1stop.ai / Personifwy** (Jul–Sep 2025)

---

## 📌 Project Overview

This project builds a binary text classification system to detect whether a given piece of text contains hate speech or not. It covers the full NLP pipeline — from raw text cleaning to model training, evaluation, and inference.

**Problem Statement:** Automatically identify and flag hate speech in online text at scale, reducing manual moderation effort.

---

## 📁 Project Structure

```
hate-speech-detection/
│
├── data/
│   ├── raw/                  # Original dataset (place here)
│   └── processed/            # Cleaned & tokenized data (auto-generated)
│
├── notebooks/
│   └── hate_speech_eda.ipynb # Exploratory Data Analysis
│
├── src/
│   ├── preprocess.py         # Text cleaning & feature extraction
│   ├── train.py              # Model training pipeline
│   ├── evaluate.py           # Metrics & evaluation
│   └── predict.py            # Inference on new text
│
├── models/
│   └── model.pkl             # Saved trained model (auto-generated)
│
├── requirements.txt
└── README.md
```

---

## 🧠 Approach

| Step | Details |
|------|---------|
| **Dataset** | Twitter Hate Speech Dataset (labeled: hate / not-hate) |
| **Preprocessing** | Lowercasing, URL/mention/punctuation removal, stopword removal, lemmatization |
| **Feature Extraction** | TF-IDF Vectorizer (unigrams + bigrams, max 10,000 features) |
| **Model** | Logistic Regression (primary), with Random Forest & SVM comparison |
| **Imbalance Handling** | SMOTE oversampling + class_weight='balanced' |
| **Evaluation** | Accuracy, Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix |

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | **92%** |
| Precision | 0.91 |
| Recall | 0.90 |
| F1-Score | 0.90 |
| AUC-ROC | 0.96 |

---

## 🚀 Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/imagam18/hate-speech-detection.git
cd hate-speech-detection
pip install -r requirements.txt
```

### 2. Add Dataset

Download the dataset and place it in `data/raw/`. Recommended:
- [Twitter Hate Speech Dataset – Kaggle](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset)

Rename the file to `data/raw/hate_speech.csv`

### 3. Train the Model

```bash
python src/train.py
```

### 4. Evaluate

```bash
python src/evaluate.py
```

### 5. Predict on New Text

```bash
python src/predict.py --text "your input text here"
```

---

## 🔧 Tech Stack

- Python 3.9+
- scikit-learn
- NLTK
- Pandas, NumPy
- Matplotlib, Seaborn
- imbalanced-learn (SMOTE)
- joblib

---

## 📜 Certificate

This project was completed as part of the **Data Science Industrial Program** at [1stop.ai](https://1stop.ai) in collaboration with Personifwy and IIT Guwahati E-Cell.

---

## 👤 Author

**Agam Saxena**  
B.Tech CSE | Lloyd Institute of Technology  
[LinkedIn](https://linkedin.com/in/agam-saxena-405645372) | [GitHub](https://github.com/imagam18)
