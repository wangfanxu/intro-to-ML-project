"""Runs stratified 10-fold cross-validation for every required model on the
training set and records AUC / accuracy (mean +/- std) for each.

Usage:
    python -m src.cv_evaluate
"""

import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import TRAIN_CSV, N_SPLITS, RANDOM_STATE, DATA_DIR
from src.pipeline import build_full_pipeline, split_X_y
from src.models import BASELINE_MODELS, ADDITIONAL_MODELS

import os

RESULTS_DIR = os.path.join(DATA_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_JSON = os.path.join(RESULTS_DIR, "cv_results.json")


def evaluate_model(name, estimator_factory, X, y, cv):
    print(f"Running 10-fold CV for {name} ...")
    start = time.time()
    pipe = build_full_pipeline(estimator_factory())
    scores = cross_validate(
        pipe, X, y, cv=cv,
        scoring=["roc_auc", "accuracy"],
        n_jobs=1,  # models already use internal n_jobs=-1
        return_train_score=False,
    )
    elapsed = time.time() - start
    result = {
        "model": name,
        "cv_auc_mean": float(np.mean(scores["test_roc_auc"])),
        "cv_auc_std": float(np.std(scores["test_roc_auc"])),
        "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
        "cv_accuracy_std": float(np.std(scores["test_accuracy"])),
        "elapsed_seconds": round(elapsed, 1),
    }
    print(
        f"  {name}: AUC={result['cv_auc_mean']:.4f}+/-{result['cv_auc_std']:.4f}, "
        f"Acc={result['cv_accuracy_mean']:.4f}+/-{result['cv_accuracy_std']:.4f} "
        f"({elapsed:.1f}s)"
    )
    return result


def main():
    train_df = pd.read_csv(TRAIN_CSV)
    X, y = split_X_y(train_df)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # NOTE: the PyTorch MLP is evaluated separately, in its own subprocess
    # (see cv_evaluate_torch.py), because on this machine importing
    # xgboost/lightgbm/catboost and then using torch's DataLoader in the
    # same process causes a native segfault (a known BLAS/OpenMP library
    # conflict on macOS ARM). Running it in an isolated process sidesteps
    # the conflict without affecting correctness of either model's results.
    all_models = {**BASELINE_MODELS, **ADDITIONAL_MODELS}

    results = []
    for name, factory in all_models.items():
        try:
            results.append(evaluate_model(name, factory, X, y, cv))
        except Exception as e:
            print(f"  FAILED: {name}: {e}")
            results.append({"model": name, "error": str(e)})

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved CV results to {RESULTS_JSON}")

    results_df = pd.DataFrame(results)
    print("\n=== CV Results Summary ===")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
