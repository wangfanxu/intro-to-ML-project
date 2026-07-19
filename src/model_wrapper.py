"""Thin wrapper around the serialized production pipeline, used by both the
Flask app and the automated test suite. Keeping this logic in a plain
Python module (rather than inline in app.py) makes it independently unit
testable.
"""

import os

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from src.config import ALL_INPUT_COLS, MODELS_DIR, TARGET_COL

DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, "production_model.joblib")


class ModelWrapper:
    """Loads a fitted pipeline and exposes prediction/evaluation helpers."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.pipeline = joblib.load(model_path)

    def predict_one(self, feature_dict: dict) -> dict:
        """Predict a single instance given as a dict of raw feature values."""
        df = pd.DataFrame([feature_dict])
        return self.predict_df(df)[0]

    def predict_df(self, df: pd.DataFrame) -> list:
        """Predict a batch given as a DataFrame of raw feature columns.

        Returns a list of dicts: [{"prediction": 0/1, "label": "goodware"/
        "malware", "probability_malware": float}, ...]
        """
        missing = [c for c in ALL_INPUT_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        X = df[ALL_INPUT_COLS]
        proba = self.pipeline.predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(int)

        return [
            {
                "prediction": int(p),
                "label": "malware" if p == 1 else "goodware",
                "probability_malware": float(prob),
            }
            for p, prob in zip(preds, proba)
        ]

    def evaluate_df(self, df: pd.DataFrame) -> dict:
        """Runs predictions plus AUC/accuracy/confusion-matrix evaluation.

        Requires the DataFrame to contain the ground-truth 'Label' column.
        """
        if TARGET_COL not in df.columns:
            raise ValueError(f"'{TARGET_COL}' column required for evaluation")

        y_true = df[TARGET_COL].astype(int).values
        X = df[ALL_INPUT_COLS]
        proba = self.pipeline.predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(int)

        return {
            "auc": float(roc_auc_score(y_true, proba)),
            "accuracy": float(accuracy_score(y_true, preds)),
            "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
            "predictions": preds.tolist(),
            "probabilities": proba.tolist(),
        }
