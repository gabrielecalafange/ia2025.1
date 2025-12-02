import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from sklearn.feature_extraction.text import HashingVectorizer  # only for type hints

DATA_DIR = Path("dataset")
INPUT_PATH = DATA_DIR / "email_spam_test.csv"   # test dataset path
MODEL_PATH = Path("spam_model_incremental.joblib")


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    package = joblib.load(MODEL_PATH)
    vectorizer: HashingVectorizer = package["vectorizer"]
    clf = package["classifier"]

    df = pd.read_csv(INPUT_PATH)

    def combine_fields(row):
        title = str(row.get("title", "") or "")
        text = str(row.get("text", "") or "")
        return f"{title} {text}"

    df["full_text"] = df.apply(combine_fields, axis=1)
    df["text_clean"] = df["full_text"].apply(clean_text)

    X_text = df["text_clean"].values
    y_true = df["type"].values

    print("Number of test samples:", len(X_text))
    print("Class distribution:")
    print(pd.Series(y_true).value_counts())

    X = vectorizer.transform(X_text)

    y_pred = clf.predict(X)

    print("Classification report:")
    print(classification_report(y_true, y_pred))

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
