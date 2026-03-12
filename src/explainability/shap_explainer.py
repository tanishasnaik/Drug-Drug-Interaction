"""
src/explainability/shap_explainer.py
--------------------------------------
Member 3 — Explainability & Validation Lead

SHAP-based explanations for DDI predictions.
Answers the question: WHY did the model predict this interaction?

LITERATURE CONTEXT:
- DeepDDI: no explainability
- MUFFIN: attention weights (limited interpretability)
- KGNN, DDIMDL: no explainability
- We use SHAP → clinicians can trust and verify predictions

Severity Scale (5-level, vs binary in prior work):
  1 - Minor          (0.0–0.2)
  2 - Moderate       (0.2–0.4)
  3 - Major          (0.4–0.6)
  4 - Severe         (0.6–0.8)
  5 - Contraindicated(0.8–1.0)

Usage:
    from src.explainability.shap_explainer import SHAPExplainer, get_severity
"""

import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Severity Configuration ────────────────────────────────────────
SEVERITY_LEVELS = [
    (0.0, 0.2, 1, "Minor",           "🟢", "green"),
    (0.2, 0.4, 2, "Moderate",        "🟡", "gold"),
    (0.4, 0.6, 3, "Major",           "🟠", "orange"),
    (0.6, 0.8, 4, "Severe",          "🔴", "red"),
    (0.8, 1.01, 5, "Contraindicated","⛔", "darkred"),
]

RECOMMENDATIONS = {
    1: "Generally safe to combine. Monitor the patient if combining long-term.",
    2: "Use with caution. Consider dose adjustment or closer monitoring.",
    3: "Clinical review strongly recommended before co-prescribing.",
    4: "Avoid this combination unless benefits clearly outweigh risks.",
    5: "Do NOT use together. Seek alternative medications immediately.",
}


def get_severity(probability: float) -> dict:
    """
    Converts a model probability (0–1) to a 5-level severity rating.
    This is our key improvement over binary prediction in prior work.

    Args:
        probability: model's predicted interaction probability

    Returns:
        dict: probability, severity_level (1-5), severity_label, emoji,
              color, recommendation
    """
    probability = float(np.clip(probability, 0.0, 1.0))
    for low, high, level, label, emoji, color in SEVERITY_LEVELS:
        if low <= probability < high:
            return {
                "probability":     round(probability, 4),
                "severity_level":  level,
                "severity_label":  label,
                "emoji":           emoji,
                "color":           color,
                "recommendation":  RECOMMENDATIONS[level],
            }
    # Fallback
    return {
        "probability": probability, "severity_level": 5,
        "severity_label": "Contraindicated", "emoji": "⛔",
        "color": "darkred", "recommendation": RECOMMENDATIONS[5]
    }


class SHAPExplainer:
    """
    Wraps SHAP TreeExplainer for our XGBoost/RandomForest models.
    Provides per-prediction explanations usable in Streamlit.
    """

    def __init__(self, model, feature_names: list = None):
        """
        Args:
            model: trained XGBoost or RandomForest model object (.model attribute)
            feature_names: list of feature name strings
        """
        self.model = model.model  # unwrap our wrapper class
        self.feature_names = feature_names
        self.explainer = None

    def fit(self, X_background: np.ndarray):
        """
        Initializes the SHAP TreeExplainer.
        X_background: a sample of training data (50-200 rows is sufficient)
        """
        log.info("Initializing SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model)
        log.info("SHAP explainer ready")

    def explain_pair(self, features: np.ndarray) -> dict:
        """
        Explains the model prediction for one drug pair.

        Args:
            features: 1D feature vector for a single drug pair

        Returns:
            dict:
              - shap_values: full SHAP value array
              - top_risk_features: top 10 features increasing interaction risk
              - top_protective_features: top 5 features decreasing risk
              - base_value: model's average prediction baseline
        """
        if self.explainer is None:
            raise RuntimeError("Call .fit(X_background) before explaining")

        x = features.reshape(1, -1)
        shap_vals = self.explainer.shap_values(x)

        # Tree models return list [class0_vals, class1_vals]
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]  # class 1 = interaction present
        else:
            sv = shap_vals[0]

        names = self.feature_names or [f"f{i}" for i in range(len(sv))]

        # Sort by absolute SHAP value
        ranked = sorted(zip(names, sv), key=lambda x: abs(x[1]), reverse=True)

        top_risk = [
            {"feature": name, "shap_value": round(float(val), 6)}
            for name, val in ranked if val > 0
        ][:10]

        top_protective = [
            {"feature": name, "shap_value": round(float(val), 6)}
            for name, val in ranked if val < 0
        ][:5]

        base_value = float(self.explainer.expected_value)
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(base_value[1])

        return {
            "shap_values":             sv,
            "top_risk_features":       top_risk,
            "top_protective_features": top_protective,
            "base_value":              round(base_value, 4),
        }

    def plot_summary(self, X: np.ndarray,
                     save_path: str = "results/figures/shap_summary.png",
                     max_display: int = 20):
        """
        Generates a SHAP beeswarm summary plot showing overall feature importance.
        Saved to results/figures/ for the project report.
        """
        if self.explainer is None:
            raise RuntimeError("Call .fit(X_background) first")

        log.info("Generating SHAP summary plot...")
        shap_vals = self.explainer.shap_values(X)
        if isinstance(shap_vals, list):
            sv = shap_vals[1]
        else:
            sv = shap_vals

        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            sv, X,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"SHAP summary saved to {save_path}")

    def plot_waterfall(self, features: np.ndarray,
                       save_path: str = "results/figures/shap_waterfall.png"):
        """
        Generates a SHAP waterfall plot for a single prediction.
        Shows exactly how each feature pushed the prediction up or down.
        """
        if self.explainer is None:
            raise RuntimeError("Call .fit(X_background) first")

        x = features.reshape(1, -1)
        explanation = shap.Explanation(
            values=self.explainer.shap_values(x)[1][0]
                   if isinstance(self.explainer.shap_values(x), list)
                   else self.explainer.shap_values(x)[0],
            base_values=self.explainer.expected_value[1]
                        if isinstance(self.explainer.expected_value, (list, np.ndarray))
                        else self.explainer.expected_value,
            data=x[0],
            feature_names=self.feature_names,
        )

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Waterfall plot saved to {save_path}")


# ── Quick test ──────────────────────────────────────────────────
if __name__ == "__main__":
    for prob in [0.1, 0.35, 0.55, 0.72, 0.91]:
        sev = get_severity(prob)
        print(f"P={prob:.2f} → {sev['emoji']} {sev['severity_label']} (Level {sev['severity_level']})")
