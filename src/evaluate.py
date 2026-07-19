"""Evaluates a trained pipeline (joblib artifact) against a CSV file.

If the CSV contains the 'Label' target column, prints/report AUC, accuracy,
and confusion matrix. Otherwise just produces predictions.

Usage:
    python -m src.evaluate --data data/test.csv --model models/production_model.joblib
"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from src.config import TARGET_COL, TEST_CSV, MODELS_DIR
import os


def evaluate(model_path: str, data_path: str) -> dict:
    pipeline = joblib.load(model_path)
    df = pd.read_csv(data_path)

    has_labels = TARGET_COL in df.columns
    if has_labels:
        X = df.drop(columns=[TARGET_COL])
        y_true = df[TARGET_COL].values
    else:
        X = df

    y_proba = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    result = {"n_rows": len(df), "predictions": y_pred.tolist(), "probabilities": y_proba.tolist()}

    if has_labels:
        auc = roc_auc_score(y_true, y_proba)
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred).tolist()
        result.update({"auc": auc, "accuracy": acc, "confusion_matrix": cm})
        print(f"AUC: {auc:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Confusion matrix (rows=true, cols=pred) [[TN, FP], [FN, TP]]:\n{np.array(cm)}")
    else:
        print(f"Generated predictions for {len(df)} rows (no ground-truth labels present).")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=TEST_CSV)
    parser.add_argument("--model", default=os.path.join(MODELS_DIR, "production_model.joblib"))
    parser.add_argument("--out", default=None, help="Optional path to save JSON results")
    args = parser.parse_args()

    result = evaluate(args.model, args.data)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {k: v for k, v in result.items() if k not in ("predictions", "probabilities")},
                f, indent=2,
            )
        print(f"Saved summary to {args.out}")


if __name__ == "__main__":
    main()
