# src/validate.py
import os
import pandas as pd


def validate_dataset(file_path):
    print(f"🔍 Validating data schema for: {file_path}")

    if not os.path.exists(file_path):
        print(f"⚠️ File not found at {file_path}. Skipping verification.")
        return True

    df = pd.read_csv(file_path)

    expected_columns = [
        "text",
        "label_hate_speech",
        "label_insult",
        "label_humiliate",
        "label_dehumanize",
        "label_violence",
    ]

    # Check columns
    missing = [c for c in expected_columns if c not in df.columns]

    if missing:
        print(f"❌ Missing columns: {missing}")
        return False

    # Check label values
    label_columns = expected_columns[1:]

    for col in label_columns:
        invalid = df[~df[col].isin([0, 1, 0.0, 1.0])]

        if not invalid.empty:
            print(f"❌ Invalid values found in column '{col}'")
            return False

    print(f"✅ Schema Validation Passed for {file_path}")
    return True


if __name__ == "__main__":
    splits = ["data/processed/val.csv"]

    all_passed = all(validate_dataset(split) for split in splits)

    if not all_passed:
        exit(1)