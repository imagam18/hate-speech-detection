"""
preprocess.py
-------------
Text cleaning and TF-IDF feature extraction for hate speech detection.
"""

import re
import string
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os

# Download required NLTK data on first run
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """
    Full NLP preprocessing pipeline:
    - Lowercase
    - Remove URLs, mentions, hashtags, punctuation, numbers
    - Remove stopwords
    - Lemmatize tokens
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)       # remove URLs
    text = re.sub(r"@\w+", "", text)                  # remove mentions
    text = re.sub(r"#\w+", "", text)                  # remove hashtags
    text = re.sub(r"\d+", "", text)                   # remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)


def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """
    Load CSV dataset and apply cleaning.
    Expected columns: 'tweet' (text) and 'class' (0=hate, 1=offensive, 2=neither)
    Binarizes: 0 (hate) → 1, others → 0
    """
    df = pd.read_csv(filepath)

    # Support both common column naming conventions
    text_col = "tweet" if "tweet" in df.columns else df.columns[0]
    label_col = "class" if "class" in df.columns else df.columns[1]

    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "label"]
    df.dropna(inplace=True)

    # Binarize: hate speech (0) = 1, everything else = 0
    df["label"] = df["label"].apply(lambda x: 1 if x == 0 else 0)
    df["clean_text"] = df["text"].apply(clean_text)

    print(f"Dataset loaded: {len(df)} rows | Class balance:\n{df['label'].value_counts()}")
    return df


def build_tfidf(train_texts, save_path: str = "models/tfidf_vectorizer.pkl"):
    """
    Fit TF-IDF vectorizer on training text. Saves to disk.
    Uses unigrams + bigrams, max 10,000 features.
    """
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10_000,
        min_df=2,
        sublinear_tf=True
    )
    X = tfidf.fit_transform(train_texts)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(tfidf, save_path)
    print(f"TF-IDF vectorizer saved → {save_path}")
    return X, tfidf


def transform_tfidf(texts, vectorizer_path: str = "models/tfidf_vectorizer.pkl"):
    """Load saved TF-IDF vectorizer and transform new texts."""
    tfidf = joblib.load(vectorizer_path)
    return tfidf.transform(texts)
