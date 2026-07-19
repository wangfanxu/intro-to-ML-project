"""Custom feature engineering transformer.

This stage runs BEFORE the ColumnTransformer (scaling/encoding/vectorizing)
and only derives features from fixed, data-independent rules (keyword
lookups, presence checks, dropping known-uninformative columns). Because
none of this depends on statistics estimated from the data, it is safe to
apply identically to train/validation/test folds without causing leakage.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import DROP_COLS, IDENTIFY_KEYWORDS, TARGET_COL


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds engineered features and drops uninformative/leaky columns.

    - Drops SHA1, FirstSeenDate, Magic, PE_TYPE (see config.DROP_COLS).
    - From 'Identify' (free-text packer/compiler signature):
        * `identify_missing`: 1 if the signature could not be determined.
        * one binary flag per keyword in IDENTIFY_KEYWORDS.
      The raw 'Identify' text column is dropped afterward (too high
      cardinality / too many missing values to encode directly).
    - From 'ImportedDlls' / 'ImportedSymbols' (space-separated token lists):
        * `num_imported_dlls`, `num_imported_symbols`: token counts.
      The raw text columns are kept so that downstream CountVectorizer
      steps (fit only on training folds) can extract common-token
      indicator features.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        if "Identify" in df.columns:
            identify_lower = df["Identify"].fillna("").str.lower()
            df["identify_missing"] = df["Identify"].isna().astype(int)
            for flag_name, keyword in IDENTIFY_KEYWORDS.items():
                df[flag_name] = identify_lower.str.contains(keyword, regex=False).astype(int)
            df = df.drop(columns=["Identify"])

        if "ImportedDlls" in df.columns:
            df["num_imported_dlls"] = (
                df["ImportedDlls"].fillna("").str.split().apply(len)
            )
        if "ImportedSymbols" in df.columns:
            df["num_imported_symbols"] = (
                df["ImportedSymbols"].fillna("").str.split().apply(len)
            )

        drop_existing = [c for c in DROP_COLS if c in df.columns]
        df = df.drop(columns=drop_existing)

        # Ensure the text columns feeding downstream CountVectorizers never
        # contain NaN (which CountVectorizer cannot handle).
        for col in ("ImportedDlls", "ImportedSymbols"):
            if col in df.columns:
                df[col] = df[col].fillna("")

        return df

    def get_feature_names_out(self, input_features=None):
        raise NotImplementedError
