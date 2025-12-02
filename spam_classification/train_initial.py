import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
import joblib

DATA_DIR = Path("dataset")
INPUT_PATH = DATA_DIR / "email_spam_phase1.csv"   # training dataset path
MODEL_PATH = Path("spam_model_incremental.joblib")


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # remove urls
    text = re.sub(r"\d+", " ", text)              # remove numbers
    text = re.sub(r"[^\w\s]", " ", text)          # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()      # remove extra spaces
    return text


def main():
    df = pd.read_csv(INPUT_PATH)

    # combine subject and body
    def combine_fields(row):
        title = str(row.get("title", "") or "")
        text = str(row.get("text", "") or "")
        return f"{title} {text}"

    df["full_text"] = df.apply(combine_fields, axis=1)
    df["text_clean"] = df["full_text"].apply(clean_text)

    X_text = df["text_clean"].values
    y = df["type"].values

    print("Number of samples for initial training:", len(X_text))
    print("Class distribution:")
    print(pd.Series(y).value_counts())

    # hashing based vectorizer (no fit required)
    vectorizer = HashingVectorizer(
        n_features=2**18,   # vector size; 2^18 = 262144 features
        alternate_sign=False,
        ngram_range=(1, 2)
    )

    X = vectorizer.transform(X_text)

    # incremental model
    clf = SGDClassifier(
        loss="log_loss",    # binary classification similar to logistic regression
        max_iter=5,
        tol=None
    )

    # need to provide classes on first partial_fit call
    classes = np.unique(y)

    print("Training model (initial)...")
    clf.partial_fit(X, y, classes=classes)
    print("Initial training completed.")

    # store vectorizer and model together
    package = {
        "vectorizer": vectorizer,
        "classifier": clf,
        "classes": classes
    }

    joblib.dump(package, MODEL_PATH)
    print("Incremental model saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()
