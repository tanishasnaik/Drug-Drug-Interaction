"""
src/models/baseline/xgboost_model.py
--------------------------------------
Member 2 — Model Development Lead

XGBoost classifier for DDI prediction.
XGBoost is our primary baseline — strong performance on tabular/fingerprint data,
and works naturally with SHAP for explainability (Member 3's module).

Usage:
    from src.models.baseline.xgboost_model import XGBoostDDIModel
"""

import numpy as np
import joblib
import xgboost as xgb
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config import get_config

log = get_logger(__name__)
model_cfg = get_config("model")["xgboost"]


class XGBoostDDIModel:
    """XGBoost wrapper for drug-drug interaction prediction."""

    def __init__(self, **kwargs):
        params = {**model_cfg, **kwargs}
        params.pop("random_state", None)  # xgb uses random_state differently
        self.model = xgb.XGBClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            eval_metric="logloss",
            use_label_encoder=False,
            seed=42,
        )
        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Train the model. Optionally pass validation data for early stopping."""
        log.info(f"Training XGBoost on {X_train.shape[0]} samples, {X_train.shape[1]} features")
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )
        self.is_trained = True
        log.info("XGBoost training complete")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns probability of interaction (class 1)."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Returns binary predictions (0 or 1)."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: str = "models/baseline/xgboost.pkl"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        log.info(f"XGBoost model saved to {path}")

    @classmethod
    def load(cls, path: str = "models/baseline/xgboost.pkl"):
        obj = cls.__new__(cls)
        obj.model = joblib.load(path)
        obj.is_trained = True
        log.info(f"XGBoost model loaded from {path}")
        return obj
