"""
src/models/baseline/logistic_regression.py
--------------------------------------------
Member 2 — Model Development Lead

Logistic Regression baseline — simplest model, most interpretable.
Used to establish a performance floor for the comparison table in the report.

Usage:
    from src.models.baseline.logistic_regression import LogisticDDIModel
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from src.utils.logger import get_logger
from src.utils.config import get_config

log = get_logger(__name__)
model_cfg = get_config("model")["logistic_regression"]


class LogisticDDIModel:
    """Logistic Regression wrapper for DDI prediction."""

    def __init__(self, **kwargs):
        params = {**model_cfg, **kwargs}
        self.model = LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 1000),
            solver=params.get("solver", "lbfgs"),
            random_state=42,
        )
        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        log.info(f"Training Logistic Regression on {X_train.shape[0]} samples")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        log.info("Logistic Regression training complete")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str = "models/baseline/logistic_regression.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        log.info(f"Logistic Regression saved to {path}")

    @classmethod
    def load(cls, path: str = "models/baseline/logistic_regression.pkl"):
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        obj.is_trained = True
        return obj
