"""Unit tests for src/model_wrapper.py.

Uses the actual trained production_model.joblib artifact and a tiny
in-memory sample of raw rows (rather than hitting the full dataset), so
these tests are fast and self-contained. If the artifact is not present
(e.g. a fresh checkout before running `python -m src.train`), the module
is skipped rather than failed.
"""

import os

import pandas as pd
import pytest

from src.config import ALL_INPUT_COLS, TARGET_COL
from src.model_wrapper import DEFAULT_MODEL_PATH, ModelWrapper

pytestmark = pytest.mark.skipif(
    not os.path.exists(DEFAULT_MODEL_PATH),
    reason="production_model.joblib not found; run `python -m src.train` first",
)


@pytest.fixture(scope="module")
def wrapper():
    return ModelWrapper()


@pytest.fixture(scope="module")
def sample_rows():
    """Two rows drawn from the small demo CSV bundled in the repo."""
    from src.config import DEMO_SAMPLE_CSV

    df = pd.read_csv(DEMO_SAMPLE_CSV)
    return df.head(5).reset_index(drop=True)


def test_predict_one_returns_expected_shape(wrapper, sample_rows):
    row = sample_rows.drop(columns=[TARGET_COL]).iloc[0].to_dict()
    result = wrapper.predict_one(row)

    assert set(result.keys()) == {"prediction", "label", "probability_malware"}
    assert result["prediction"] in (0, 1)
    assert result["label"] in ("goodware", "malware")
    assert 0.0 <= result["probability_malware"] <= 1.0


def test_predict_df_batch(wrapper, sample_rows):
    X = sample_rows.drop(columns=[TARGET_COL])
    results = wrapper.predict_df(X)

    assert len(results) == len(sample_rows)
    for r in results:
        assert r["prediction"] in (0, 1)


def test_predict_df_raises_on_missing_columns(wrapper, sample_rows):
    X = sample_rows.drop(columns=[TARGET_COL, "Entropy"])
    with pytest.raises(ValueError, match="Missing required columns"):
        wrapper.predict_df(X)


def test_evaluate_df_returns_metrics(wrapper, sample_rows):
    metrics = wrapper.evaluate_df(sample_rows)

    assert 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 2
    assert len(metrics["predictions"]) == len(sample_rows)


def test_evaluate_df_requires_label_column(wrapper, sample_rows):
    X = sample_rows.drop(columns=[TARGET_COL])
    with pytest.raises(ValueError, match="Label"):
        wrapper.evaluate_df(X)


def test_all_input_cols_present_in_demo_sample(sample_rows):
    for col in ALL_INPUT_COLS:
        assert col in sample_rows.columns
