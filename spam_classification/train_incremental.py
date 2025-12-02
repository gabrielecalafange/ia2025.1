import argparse
import re
from pathlib import Path

import pandas as pd
import joblib

from sklearn.feature_extraction.text import HashingVectorizer  # type hints only


DATA_DIR = Path("dataset")
DEFAULT_PHASE = 5
DEFAULT_INPUT_PATH = DATA_DIR / f"email_spam_phase{DEFAULT_PHASE}.csv"
MODEL_PATH = Path("spam_model_incremental.joblib")


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def run_incremental_update(input_path: Path, model_path: Path = MODEL_PATH) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    if not model_path.is_file():
        raise FileNotFoundError(f"Model package not found: {model_path}")

    # Load previously trained model
    package = joblib.load(model_path)
    vectorizer: HashingVectorizer = package["vectorizer"]
    clf = package["classifier"]
    classes = package.get("classes")

    df = pd.read_csv(input_path)

    def combine_fields(row):
        title = str(row.get("title", "") or "")
        text = str(row.get("text", "") or "")
        return f"{title} {text}"

    df["full_text"] = df.apply(combine_fields, axis=1)
    df["text_clean"] = df["full_text"].apply(clean_text)

    X_text = df["text_clean"].values
    y = df["type"].values

    print("New samples for incremental training:", len(X_text))
    print("Class distribution in the new batch:")
    print(pd.Series(y).value_counts())

    # Transform text into numbers with the SAME vectorizer
    X = vectorizer.transform(X_text)

    print("Updating model with new data...")
    clf.partial_fit(X, y)   # no need to pass classes again here
    print("Incremental training completed.")

    # Update the package and persist it
    package["classifier"] = clf
    joblib.dump(package, model_path)
    print("Updated model saved to:", model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an incremental training step using a specific dataset phase.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"CSV path for the incremental batch (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--phase",
        type=int,
        help="If provided, overrides --data-path with dataset/email_spam_phase{phase}.csv",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help=f"Path to the incremental model package (default: {MODEL_PATH})",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = args.data_path
    if args.phase is not None:
        data_path = DATA_DIR / f"email_spam_phase{args.phase}.csv"
    run_incremental_update(data_path, args.model_path)


if __name__ == "__main__":
    main()
