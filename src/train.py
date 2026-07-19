"""Trains the final production model (best model from CV) on the full
training set and serializes the fitted pipeline to disk.

Usage:
    python -m src.train
"""

import os
import joblib
import pandas as pd

from src.config import TRAIN_CSV, MODELS_DIR, RANDOM_STATE
from src.pipeline import build_full_pipeline, split_X_y
from src.models import lightgbm_clf

MODEL_PATH = os.path.join(MODELS_DIR, "production_model.joblib")

# Selected based on 10-fold CV results (see results/cv_results.json /
# evaluation-and-design.md): LightGBM had the highest mean CV AUC (0.9984)
# with the lowest variance across folds.
BEST_MODEL_NAME = "lightgbm"
BEST_MODEL_FACTORY = lightgbm_clf


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    train_df = pd.read_csv(TRAIN_CSV)
    X_train, y_train = split_X_y(train_df)

    pipeline = build_full_pipeline(BEST_MODEL_FACTORY())
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Trained '{BEST_MODEL_NAME}' on {len(X_train)} rows.")
    print(f"Saved production pipeline to {MODEL_PATH}")


if __name__ == "__main__":
    main()
