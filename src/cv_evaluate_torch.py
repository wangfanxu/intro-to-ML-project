"""Runs stratified 10-fold cross-validation for the PyTorch MLP baseline in
its own process. Kept separate from cv_evaluate.py because importing
xgboost/lightgbm/catboost and then using torch's DataLoader in the same
process causes a native segfault on this machine (BLAS/OpenMP conflict on
macOS ARM). Running in isolation sidesteps the conflict.

Usage:
    python -m src.cv_evaluate_torch
"""

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import TRAIN_CSV, N_SPLITS, RANDOM_STATE, DATA_DIR
from src.pipeline import build_full_pipeline, split_X_y
from src.torch_mlp import TorchMLPClassifier

RESULTS_DIR = os.path.join(DATA_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_JSON = os.path.join(RESULTS_DIR, "cv_results_torch.json")


def main():
    train_df = pd.read_csv(TRAIN_CSV)
    X, y = split_X_y(train_df)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    print("Running 10-fold CV for pytorch_mlp ...")
    start = time.time()
    pipe = build_full_pipeline(
        TorchMLPClassifier(hidden_dims=(64, 32), epochs=15, lr=1e-3)
    )
    scores = cross_validate(
        pipe, X, y, cv=cv,
        scoring=["roc_auc", "accuracy"],
        n_jobs=1,
        return_train_score=False,
    )
    elapsed = time.time() - start
    result = {
        "model": "pytorch_mlp",
        "cv_auc_mean": float(np.mean(scores["test_roc_auc"])),
        "cv_auc_std": float(np.std(scores["test_roc_auc"])),
        "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
        "cv_accuracy_std": float(np.std(scores["test_accuracy"])),
        "elapsed_seconds": round(elapsed, 1),
    }
    print(
        f"  pytorch_mlp: AUC={result['cv_auc_mean']:.4f}+/-{result['cv_auc_std']:.4f}, "
        f"Acc={result['cv_accuracy_mean']:.4f}+/-{result['cv_accuracy_std']:.4f} "
        f"({elapsed:.1f}s)"
    )

    with open(RESULTS_JSON, "w") as f:
        json.dump([result], f, indent=2)
    print(f"Saved to {RESULTS_JSON}")


if __name__ == "__main__":
    main()
