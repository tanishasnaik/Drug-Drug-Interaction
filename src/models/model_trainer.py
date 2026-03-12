"""
src/models/model_trainer.py
-----------------------------
Member 2 — Model Development Lead

Unified training pipeline that:
  - Splits data into train/val/test
  - Trains all baseline models
  - Saves models and metrics
  - Runs cross-validation

Usage:
    from src.models.model_trainer import ModelTrainer
    trainer = ModelTrainer()
    results = trainer.train_all(X, y, feature_names)
"""

import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from src.models.baseline.xgboost_model import XGBoostDDIModel
from src.models.baseline.random_forest import RandomForestDDIModel
from src.models.baseline.logistic_regression import LogisticDDIModel
from src.utils.logger import get_logger
from src.utils.config import get_config

log = get_logger(__name__)
train_cfg = get_config("model")["training"]


class ModelTrainer:
    """Handles training, validation, and saving of all DDI models."""

    def __init__(self):
        self.models = {
            "logistic_regression": LogisticDDIModel(),
            "random_forest":       RandomForestDDIModel(),
            "xgboost":             XGBoostDDIModel(),
        }
        self.results = {}

    def train_all(self, X: np.ndarray, y: np.ndarray,
                  feature_names: list = None) -> dict:
        """
        Trains all three baseline models and returns evaluation results.

        Args:
            X: feature matrix
            y: labels
            feature_names: optional list of feature names for explainability

        Returns:
            dict of model_name → metrics dict
        """
        log.info(f"Starting training pipeline — {X.shape[0]} samples, {X.shape[1]} features")

        # Train / test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=train_cfg["cross_validation_folds"] * 0.04,  # ~20%
            random_state=42,
            stratify=y
        )
        log.info(f"Train: {len(X_train)} | Test: {len(X_test)}")
        log.info(f"Class balance — Train: {np.bincount(y_train)} | Test: {np.bincount(y_test)}")

        for name, model in self.models.items():
            log.info(f"\n{'='*40}")
            log.info(f"Training: {name}")
            model.train(X_train, y_train)
            metrics = self._evaluate(model, X_test, y_test)
            self.results[name] = metrics
            model.save(f"models/baseline/{name.replace('_', '_')}.pkl")
            log.info(f"{name} → AUROC: {metrics['auroc']}, F1: {metrics['f1']}")

        # Save metrics
        self._save_metrics()
        self._print_comparison()
        return self.results

    def _evaluate(self, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluates a model and returns metric dict."""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        return {
            "auroc":     round(roc_auc_score(y_test, y_prob), 4),
            "f1":        round(f1_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred), 4),
        }

    def _save_metrics(self):
        """Saves all model metrics to results/metrics/"""
        Path("results/metrics").mkdir(parents=True, exist_ok=True)
        with open("results/metrics/baseline_metrics.json", "w") as f:
            json.dump(self.results, f, indent=2)
        log.info("Metrics saved to results/metrics/baseline_metrics.json")

    def _print_comparison(self):
        """Prints a formatted model comparison table."""
        log.info("\n📊 Model Comparison Table")
        log.info(f"{'Model':<25} {'AUROC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8}")
        log.info("-" * 65)
        for name, metrics in self.results.items():
            log.info(
                f"{name:<25} {metrics['auroc']:>8} {metrics['f1']:>8} "
                f"{metrics['precision']:>10} {metrics['recall']:>8}"
            )


# ── Entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick test with dummy data
    np.random.seed(42)
    X_dummy = np.random.randint(0, 2, (300, 4139))
    y_dummy = np.random.randint(0, 2, (300,))

    trainer = ModelTrainer()
    results = trainer.train_all(X_dummy, y_dummy)
