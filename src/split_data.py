"""Load raw dataset, deduplicate, and produce a stratified 80/20 train/test
split BEFORE any preprocessing or feature engineering to prevent data leakage.

Run as a script:
    python -m src.split_data
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RAW_CSV, TRAIN_CSV, TEST_CSV, TARGET_COL, TEST_SIZE, RANDOM_STATE


def load_and_dedupe(path: str = RAW_CSV) -> pd.DataFrame:
    """Load the raw CSV and remove duplicate rows / duplicate file hashes.

    The raw dataset contains exact duplicate rows and repeated SHA1 hashes
    (the same executable appearing more than once). Both are removed here,
    prior to splitting, so that the same file cannot appear in both the
    train and test partitions (which would leak information and inflate
    test performance).
    """
    df = pd.read_csv(path)
    n_before = len(df)

    df = df.drop_duplicates()
    n_after_dedupe_rows = len(df)

    if "SHA1" in df.columns:
        df = df.drop_duplicates(subset=["SHA1"], keep="first")
    n_after_dedupe_sha1 = len(df)

    print(f"Rows before dedup: {n_before}")
    print(f"Rows after dropping exact duplicate rows: {n_after_dedupe_rows}")
    print(f"Rows after dropping duplicate SHA1 hashes: {n_after_dedupe_sha1}")

    return df.reset_index(drop=True)


def split(df: pd.DataFrame):
    """Stratified 80/20 split by target label."""
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df[TARGET_COL],
        random_state=RANDOM_STATE,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def main():
    df = load_and_dedupe()
    train_df, test_df = split(df)

    print(f"\nTrain shape: {train_df.shape}, Label balance:\n{train_df[TARGET_COL].value_counts(normalize=True)}")
    print(f"\nTest shape: {test_df.shape}, Label balance:\n{test_df[TARGET_COL].value_counts(normalize=True)}")

    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    print(f"\nSaved train set to {TRAIN_CSV}")
    print(f"Saved test set to {TEST_CSV}")


if __name__ == "__main__":
    main()
