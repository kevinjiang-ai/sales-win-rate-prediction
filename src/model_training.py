"""
Train Logistic Regression (baseline) and Gradient Boosting models.
Saves trained models to models/ directory.
"""

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

LR_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": 42,
}

GB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_samples_leaf": 20,
    "subsample": 0.8,
    "random_state": 42,
}


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    print("Training Logistic Regression...")
    model = LogisticRegression(**LR_PARAMS)
    model.fit(X_train, y_train)
    _save_model(model, "logistic_regression.pkl")
    return model


def train_gradient_boosting(X_train, y_train) -> GradientBoostingClassifier:
    print("Training Gradient Boosting...")
    model = GradientBoostingClassifier(**GB_PARAMS)
    model.fit(X_train, y_train)
    _save_model(model, "gradient_boosting.pkl")
    return model


def _save_model(model, filename: str) -> None:
    path = MODELS_DIR / filename
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved → {path}")


def load_model(filename: str):
    path = MODELS_DIR / filename
    with open(path, "rb") as f:
        return pickle.load(f)
