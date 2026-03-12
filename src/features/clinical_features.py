"""
src/features/clinical_features.py
-----------------------------------
Member 2 — Model Development Lead

Encodes real-world clinical risk signals from FDA FAERS data as features.

LITERATURE CONTEXT:
- NONE of the prior works (DeepDDI, MUFFIN, KGNN, DDIMDL) used real-world data
- DrugBank is static — doesn't reflect actual clinical outcomes
- We integrate FDA adverse event frequency as a continuous risk feature

Clinical features per drug pair:
  - fda_report_count_a: how many adverse reports involve drug A
  - fda_report_count_b: how many adverse reports involve drug B
  - fda_risk_score: normalized combined risk signal

Usage:
    from src.features.clinical_features import build_clinical_matrix
"""

import numpy as np
import pandas as pd
from src.utils.logger import get_logger

log = get_logger(__name__)


def normalize_report_count(count: float, max_count: float = 1000.0) -> float:
    """Normalizes an FDA report count to [0, 1] range."""
    return min(float(count) / max_count, 1.0)


def build_clinical_vector(report_count_a: int, report_count_b: int) -> np.ndarray:
    """
    Builds a clinical feature vector for a drug pair.

    Features (3 dimensions):
      [0] normalized report count for drug A
      [1] normalized report count for drug B
      [2] combined risk signal (geometric mean of A and B)

    Returns:
        np.ndarray of shape (3,)
    """
    norm_a = normalize_report_count(report_count_a)
    norm_b = normalize_report_count(report_count_b)
    combined = (norm_a * norm_b) ** 0.5   # Geometric mean

    return np.array([norm_a, norm_b, combined], dtype=np.float32)


def build_clinical_matrix(df_pairs: pd.DataFrame,
                           df_clinical: pd.DataFrame) -> tuple:
    """
    Builds the clinical feature matrix for all drug pairs.

    Args:
        df_pairs: DataFrame with columns drug_a, drug_b
        df_clinical: DataFrame with columns drug_name, fda_report_count

    Returns:
        X_clinical: np.ndarray (n_pairs, 3)
        feature_names: list of feature name strings
    """
    report_map = df_clinical.set_index("drug_name")["fda_report_count"].to_dict()

    X_clinical = []
    for _, row in df_pairs.iterrows():
        count_a = report_map.get(row["drug_a"], 0)
        count_b = report_map.get(row["drug_b"], 0)
        vec = build_clinical_vector(count_a, count_b)
        X_clinical.append(vec)

    X_clinical = np.array(X_clinical)
    feature_names = ["fda_norm_a", "fda_norm_b", "fda_combined_risk"]

    log.info(f"Clinical feature matrix shape: {X_clinical.shape}")
    return X_clinical, feature_names


# ── Quick test ──────────────────────────────────────────────────
if __name__ == "__main__":
    vec = build_clinical_vector(report_count_a=500, report_count_b=200)
    print("Clinical feature vector:", vec)
