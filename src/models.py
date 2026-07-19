"""Model factory functions for all required baseline + additional models.

Hyperparameters are fixed, reasonable defaults (light manual tuning) rather
than exhaustive grid search, to keep 10-fold CV runtime tractable on the
~35k-row training set within this project's time budget. This is documented
in evaluation-and-design.md.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.config import RANDOM_STATE


def logistic_regression():
    return LogisticRegression(
        max_iter=2000,
        C=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def decision_tree():
    return DecisionTreeClassifier(
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def xgboost_clf():
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="auc",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def lightgbm_clf():
    return LGBMClassifier(
        n_estimators=300,
        max_depth=-1,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=-1,
    )


def catboost_clf():
    return CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        verbose=False,
    )


BASELINE_MODELS = {
    "logistic_regression": logistic_regression,
    "decision_tree": decision_tree,
    "random_forest": random_forest,
}

ADDITIONAL_MODELS = {
    "xgboost": xgboost_clf,
    "lightgbm": lightgbm_clf,
    "catboost": catboost_clf,
}
