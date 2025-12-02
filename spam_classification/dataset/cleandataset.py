import sys
import csv
import re
from pathlib import Path

import pandas as pd


def detect_separator(path: Path, sample_lines: int = 5) -> str:
    """
    Detect whether the CSV uses ',' or ';' as the delimiter via csv.Sniffer.
    Defaults to ',' if detection fails.
    """
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        sample = "".join([next(f) for _ in range(sample_lines)])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";"])
        sep = dialect.delimiter
        print(f"[INFO] Detected separator: '{sep}'")
        return sep
    except Exception:
        print("[WARN] Could not detect the separator automatically, using ','")
        return ","


def normalize_type(value: str) -> str | None:
    """
    Normalize the 'type' field to 'spam' or 'not spam'.
    Also accepts '1' (spam) and '0' (not spam).
    """
    if pd.isna(value):
        return None

    # Convert to string and trim spaces (handles int 0 or 1)
    v = str(value).strip().lower()

    # --- Numeric dataset adaptation ---
    if v == "1":
        return "spam"
    if v == "0":
        return "not spam"
    # ---------------------------------------

    if v in ["spam", "spams"]:
        return "spam"
    if v in ["not spam", "ham", "legit", "normal"]:
        return "not spam"
    return None  # unknown


def clean_text(text: str) -> str:
    """
    Basic text cleaning:
    - ensure the value is a string
    - remove internal line breaks
    - strip control characters
    - collapse repeated spaces
    """
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    # Replace line breaks with spaces
    text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")

    # Remove control characters (e.g., \x0b, \x0c)
    text = re.sub(r"[\x00-\x1F\x7F]", " ", text)

    # Collapse repeated spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    if len(sys.argv) < 2:
        print("Usage: python clean_dataset.py PATH_TO_CSV")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.is_file():
        print(f"[ERROR] File not found: {input_path}")
        sys.exit(1)

    sep = detect_separator(input_path)

    print(f"[INFO] Reading original CSV: {input_path}")
    # Using engine='python' helps with tougher CSV files
    df = pd.read_csv(
        input_path,
        sep=sep,
        engine="python",
        on_bad_lines="warn" 
    )

    print("[INFO] Columns found:", df.columns.tolist())

    # Try to map/rename columns if they come with other names
    columns_lower = {c.lower(): c for c in df.columns}

    # Include Portuguese synonyms to stay compatible with legacy exports
    possible_title = ["title", "subject", "assunto"]
    possible_text = ["text", "body", "mensagem", "conteudo", "content", "email_content"]
    possible_type = ["type", "label", "classe", "category"]

    def find_column(possible):
        for name in possible:
            if name in columns_lower:
                return columns_lower[name]
        return None

    col_title = find_column(possible_title)
    col_text = find_column(possible_text)
    col_type = find_column(possible_type)

    # --- Adaptation: title is optional now ---
    if not col_text or not col_type:
        print("[ERROR] Could not identify columns for 'text' and 'type/label'.")
        print("       File columns:", df.columns.tolist())
        sys.exit(1)

    print(f"[INFO] Identified columns -> Text: '{col_text}' | Class: '{col_type}'")

    if col_title:
        print(f"[INFO] Title column found: '{col_title}'")
        df = df[[col_title, col_text, col_type]].copy()
        df.columns = ["title", "text", "type"]
    else:
        print("[WARN] No title column found. Creating empty 'title' column.")
        df = df[[col_text, col_type]].copy()
        df.columns = ["text", "type"]
        df["title"] = "" # fill with empty string
    # ------------------------------------------

    # Clean text fields
    print("[INFO] Cleaning text columns...")
    df["title"] = df["title"].apply(clean_text)
    df["text"] = df["text"].apply(clean_text)

    # Normalize labels
    print("[INFO] Normalizing labels (type)...")
    df["type_norm"] = df["type"].apply(normalize_type)

    before_count = len(df)
    df = df[~df["type_norm"].isna()].copy()
    df["type"] = df["type_norm"]
    df = df.drop(columns=["type_norm"])
    after_count = len(df)

    if after_count < before_count:
        print(f"[WARN] {before_count - after_count} rows removed due to invalid labels in 'type'.")

    # Remove rows where TEXT is empty (empty titles are fine)
    empty_mask = (df["text"].str.len() == 0)
    removed_empty = empty_mask.sum()
    if removed_empty > 0:
        df = df[~empty_mask].copy()
        print(f"[WARN] {removed_empty} rows removed because 'text' was empty.")

    # Shuffle a bit to remove any biased ordering
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Ensure string types
    df["title"] = df["title"].astype(str)
    df["text"] = df["text"].astype(str)
    df["type"] = df["type"].astype(str)

    # Final stats
    print("[INFO] Final size after cleaning:", len(df))
    print("[INFO] Class distribution:")
    print(df["type"].value_counts())

    # Output path: same directory + _clean suffix
    output_path = input_path.with_name(input_path.stem + "_clean.csv")
    print(f"[INFO] Saving cleaned CSV to: {output_path}")

    # Save with comma separator and automatic quoting
    df.to_csv(output_path, index=False)

    print("[OK] Cleaning completed.")


if __name__ == "__main__":
    main()
