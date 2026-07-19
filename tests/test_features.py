"""Unit tests for the FeatureEngineer transformer (src/features.py).

These tests check the data-independent engineered-feature logic in
isolation, without touching the full model pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from src.config import DROP_COLS, IDENTIFY_KEYWORDS
from src.features import FeatureEngineer


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "SHA1": ["abc123", "def456"],
            "FirstSeenDate": ["2020-01-01", "2020-02-02"],
            "Magic": [332, 332],
            "PE_TYPE": ["PE32", "PE32"],
            "Identify": ["UPX 3.0 packer", None],
            "ImportedDlls": ["kernel32.dll user32.dll", "advapi32.dll"],
            "ImportedSymbols": ["getprocaddress loadlibrarya", "regopenkeyexa"],
            "Entropy": [6.5, 5.1],
            "Size": [1024, 2048],
        }
    )


def test_drops_uninformative_and_leaky_columns(sample_df):
    out = FeatureEngineer().fit_transform(sample_df)
    for col in DROP_COLS:
        assert col not in out.columns


def test_identify_keyword_flags_created(sample_df):
    out = FeatureEngineer().fit_transform(sample_df)
    assert "Identify" not in out.columns
    for flag_name in IDENTIFY_KEYWORDS:
        assert flag_name in out.columns

    # Row 0 has "UPX 3.0 packer" -> has_upx flag should be 1.
    assert out.loc[0, "has_upx"] == 1
    # Row 1 has missing Identify -> identify_missing flag should be 1.
    assert out.loc[1, "identify_missing"] == 1
    assert out.loc[0, "identify_missing"] == 0


def test_token_counts_and_nan_filled(sample_df):
    out = FeatureEngineer().fit_transform(sample_df)
    assert out.loc[0, "num_imported_dlls"] == 2
    assert out.loc[1, "num_imported_dlls"] == 1
    assert out.loc[0, "num_imported_symbols"] == 2

    # Raw text columns are retained (for downstream CountVectorizer) and
    # must never contain NaN.
    assert "ImportedDlls" in out.columns
    assert out["ImportedDlls"].isna().sum() == 0
    assert out["ImportedSymbols"].isna().sum() == 0


def test_handles_missing_optional_columns_gracefully():
    # If Identify/ImportedDlls/ImportedSymbols are absent, transform should
    # not raise and should simply skip those derivations.
    df = pd.DataFrame({"Entropy": [1.0], "Size": [10]})
    out = FeatureEngineer().fit_transform(df)
    assert "Entropy" in out.columns
    assert "num_imported_dlls" not in out.columns


def test_output_row_count_preserved(sample_df):
    out = FeatureEngineer().fit_transform(sample_df)
    assert len(out) == len(sample_df)
