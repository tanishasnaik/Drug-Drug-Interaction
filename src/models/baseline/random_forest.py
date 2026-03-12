"""
src/models/baseline/random_forest.py
--------------------------------------
Member 2 — Model Development Lead

Random Forest classifier for DDI prediction.
Used as a baseline comparison model against XGBoost and the GNN.

Usage:
    from src.models.baseline.random_forest import RandomForestDDIModel
"""

import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from src.utils.logger import get_logger
from src.utils.config import get_config

log = get_logger(__name__)
model_cfg = get_config("model")["random_forest"]


class RandomForestDDIModel:
    """Random Forest wrapper for DDI prediction."""

    def __init__(self, **kwargs):
        params = {**model_cfg, **kwargs}
        self.model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 10),
            min_samples_split=params.get("min_samples_split", 5),
            n_jobs=-1,
            random_state=42,
        )
        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs):
        log.info(f"Training Random Forest on {X_train.shape[0]} samples")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        log.info("Random Forest training complete")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str = "models/baseline/random_forest.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        log.info(f"Random Forest saved to {path}")

    @classmethod
    def load(cls, path: str = "models/baseline/random_forest.pkl"):
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        obj.is_trained = True
        log.info(f"Random Forest loaded from {path}")
        return obj
