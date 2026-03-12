"""
src/features/biological_features.py
-------------------------------------
Member 2 — Model Development Lead

Generates biological features for drug pairs using enzyme/target data.

LITERATURE CONTEXT:
- MUFFIN (2021) added target proteins + pathways → improved accuracy over DeepDDI
- KGNN (2020) built a full knowledge graph of drug-enzyme relationships
- We encode enzyme overlap as a compact feature vector (simpler than full KG)

The key insight: if Drug A and Drug B are BOTH metabolized by CYP2C9,
they compete for the same metabolic enzyme → higher interaction risk.

Usage:
    from src.features.biological_features import build_bio_feature_vector, build_bio_matrix
"""

import numpy as np
import pandas as pd
from src.utils.logger import get_logger

log = get_logger(__name__)

# All CYP450 enzymes and other major drug metabolizers
# These are one-hot encoded as binary features
CYP_ENZYMES = [
    "CYP1A2", "CYP2B6", "CYP2C8", "CYP2C9",
    "CYP2C19", "CYP2D6", "CYP2E1", "CYP3A4", "CYP3A5",
    "UGT1A1", "UGT1A4", "UGT2B7",
    "P-glycoprotein",
]


def enzymes_to_vector(enzyme_list: list) -> np.ndarray:
    """
    Encodes a list of enzyme names as a binary one-hot vector.
    Length = len(CYP_ENZYMES).

    Example: ["CYP2C9", "CYP3A4"] → [0, 0, 0, 1, 0, 0, 0, 1, 0, ...]
    """
    vec = np.zeros(len(CYP_ENZYMES), dtype=np.int8)
    for i, enzyme in enumerate(CYP_ENZYMES):
        if enzyme in enzyme_list:
            vec[i] = 1
    return vec


def build_bio_feature_vector(enzymes_a: list, enzymes_b: list) -> np.ndarray:
    """
    Builds a biological feature vector for a drug PAIR.

    Features included:
      - one-hot vector for drug A's enzymes     (13 dims)
      - one-hot vector for drug B's enzymes     (13 dims)
      - element-wise AND of both (shared)       (13 dims)
      - enzyme overlap count                    (1 scalar)
    Total: 40 dimensions

    Args:
        enzymes_a: list of enzyme names for drug A
        enzymes_b: list of enzyme names for drug B

    Returns:
        np.ndarray of shape (40,)
    """
    vec_a   = enzymes_to_vector(enzymes_a)
    vec_b   = enzymes_to_vector(enzymes_b)
    overlap = vec_a & vec_b                            # shared enzymes (bitwise AND)
    count   = np.array([overlap.sum()], dtype=np.float32)
    return np.concatenate([vec_a, vec_b, overlap, count])


def build_bio_matrix(df_pairs: pd.DataFrame,
                      df_biological: pd.DataFrame) -> tuple:
    """
    Builds biological feature matrix for all drug pairs.

    Args:
        df_pairs: DataFrame with columns drug_a, drug_b
        df_biological: DataFrame with columns drug_name, enzymes (pipe-separated)

    Returns:
        X_bio: np.ndarray (n_pairs, 40)
        feature_names: list of feature name strings
    """
    # Build enzyme lookup: drug_name → list of enzymes
    enzyme_map = {}
    for _, row in df_biological.iterrows():
        enzymes_str = row.get("enzymes", "")
        enzyme_map[row["drug_name"]] = (
            [e for e in enzymes_str.split("|") if e]
            if isinstance(enzymes_str, str) else []
        )

    X_bio = []
    for _, row in df_pairs.iterrows():
        enzymes_a = enzyme_map.get(row["drug_a"], [])
        enzymes_b = enzyme_map.get(row["drug_b"], [])
        vec = build_bio_feature_vector(enzymes_a, enzymes_b)
        X_bio.append(vec)

    X_bio = np.array(X_bio)
    feature_names = (
        [f"a_{e}" for e in CYP_ENZYMES] +
        [f"b_{e}" for e in CYP_ENZYMES] +
        [f"shared_{e}" for e in CYP_ENZYMES] +
        ["enzyme_overlap_count"]
    )

    log.info(f"Biological feature matrix shape: {X_bio.shape}")
    return X_bio, feature_names


def get_mechanism_text(enzymes_a: list, enzymes_b: list,
                        drug_a: str, drug_b: str) -> str:
    """
    Generates a human-readable mechanism explanation for the Streamlit UI.
    This is a key innovation: we explain WHY the interaction happens.

    Example output:
      "Warfarin and Fluoxetine are both metabolized by CYP2C9.
       Co-administration may cause competitive inhibition, leading to
       elevated plasma levels and increased bleeding risk."
    """
    shared = set(enzymes_a) & set(enzymes_b)
    if not shared:
        return (f"No shared enzyme metabolism detected between {drug_a} and {drug_b}. "
                f"Interaction risk may still exist through other mechanisms.")

    shared_str = ", ".join(sorted(shared))
    return (
        f"⚠️ {drug_a.capitalize()} and {drug_b.capitalize()} are both metabolized "
        f"by {shared_str}. Co-administration may lead to competitive inhibition, "
        f"resulting in elevated plasma levels of one or both drugs. "
        f"This can increase the risk of adverse effects or toxicity."
    )


# ── Quick test ──────────────────────────────────────────────────
if __name__ == "__main__":
    # Warfarin (CYP2C9) + Fluoxetine (CYP2D6, CYP2C9) → shared: CYP2C9
    text = get_mechanism_text(
        ["CYP2C9", "CYP3A4"],
        ["CYP2D6", "CYP2C9"],
        "warfarin", "fluoxetine"
    )
    print(text)

    vec = build_bio_feature_vector(["CYP2C9", "CYP3A4"], ["CYP2D6", "CYP2C9"])
    print(f"Bio feature vector shape: {vec.shape}")
    print(f"Overlap count: {vec[-1]}")
