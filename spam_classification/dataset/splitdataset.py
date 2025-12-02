import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    if len(sys.argv) < 2:
        print("Usage: python splitdataset.py PATH_TO_CSV")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.is_file():
        print(f"[ERROR] File not found: {input_path}")
        sys.exit(1)

    # Read the CSV (already cleaned, only splitting here)
    df = pd.read_csv(input_path)
    print("[INFO] Rows loaded:", len(df))
    print(df["type"].value_counts())

    # Validate required columns
    if not {"title", "text", "type"}.issubset(df.columns):
        print("[ERROR] CSV must contain columns: title, text, type")
        sys.exit(1)

    # 1) Hold out test set first (20%), stratified
    df_trainval, df_test = train_test_split(
        df,
        test_size=0.20,
        stratify=df["type"],
        random_state=42
    )

    print("\n[INFO] TEST (20%):", len(df_test))
    print(df_test["type"].value_counts())

    # 2) Work with the remaining 80%
    df_remaining = df_trainval.copy()

    # Separate spam and not spam
    df_spam = df_remaining[df_remaining["type"] == "spam"].sample(frac=1, random_state=42)
    df_ham = df_remaining[df_remaining["type"] != "spam"].sample(frac=1, random_state=42)

    n_rest = len(df_remaining)
    n_spam_rest = len(df_spam)
    n_ham_rest = len(df_ham)

    print("\n[INFO] REMAINING (for phases 1–5):", n_rest)
    print("spam:", n_spam_rest, "not spam:", n_ham_rest)

    # Percentages below refer to df_remaining (80% of the total)
    phase_cfg = {
        2: {"spam": 0.10, "ham": 0.05},
        3: {"spam": 0.01, "ham": 0.14},
        4: {"spam": 0.07, "ham": 0.08},
        5: {"spam": 0.03, "ham": 0.12},
    }

    phases = {}
    spam_rem = df_spam.copy()
    ham_rem = df_ham.copy()

    # Build phases 2 to 5 first
    for phase, cfg in phase_cfg.items():
        n_spam_req = int(round(cfg["spam"] * n_rest))
        n_ham_req = int(round(cfg["ham"] * n_rest))

        if n_spam_req > len(spam_rem) or n_ham_req > len(ham_rem):
            raise ValueError(f"[ERROR] Not enough spam/ham samples for phase {phase}")

        df_phase_spam = spam_rem.iloc[:n_spam_req].copy()
        spam_rem = spam_rem.iloc[n_spam_req:]

        df_phase_ham = ham_rem.iloc[:n_ham_req].copy()
        ham_rem = ham_rem.iloc[n_ham_req:]

        df_phase = pd.concat([df_phase_spam, df_phase_ham], ignore_index=True)
        df_phase = df_phase.sample(frac=1, random_state=42).reset_index(drop=True)

        phases[phase] = df_phase

        print(f"\n[INFO] Phase {phase}: {len(df_phase)} rows "
              f"(spam={len(df_phase_spam)}, ham={len(df_phase_ham)})")

    # Now build phase 1 = 50% spam + 50% not spam
    n_spam_left = len(spam_rem)
    n_ham_left = len(ham_rem)

    print("\n[INFO] Remaining for Phase 1:")
    print("spam:", n_spam_left, "ham:", n_ham_left)

    n_each = min(n_spam_left, n_ham_left)

    if n_each == 0:
        raise ValueError("[ERROR] Not enough spam/ham for a balanced phase 1")

    df_phase1_spam = spam_rem.iloc[:n_each]
    df_phase1_ham = ham_rem.iloc[:n_each]

    df_phase1 = pd.concat([df_phase1_spam, df_phase1_ham], ignore_index=True)
    df_phase1 = df_phase1.sample(frac=1, random_state=42).reset_index(drop=True)

    phases[1] = df_phase1

    print(f"\n[INFO] Phase 1: {len(df_phase1)} rows "
          f"(spam={len(df_phase1_spam)}, ham={len(df_phase1_ham)})")

    # Save files
    outdir = input_path.parent
    for phase in [1, 2, 3, 4, 5]:
        out_path = outdir / f"email_spam_phase{phase}.csv"
        phases[phase].to_csv(out_path, index=False)
        print("[SAVED]", out_path)

    test_path = outdir / "email_spam_test.csv"
    df_test.to_csv(test_path, index=False)
    print("[SAVED]", test_path)

    print("\n[OK] Split complete.")


if __name__ == "__main__":
    main()
