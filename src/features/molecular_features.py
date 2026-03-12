"""
src/features/molecular_features.py
------------------------------------
Member 2 — Model Development Lead

Generates molecular fingerprints from SMILES strings.
These are the chemical modality features — extending DeepDDI (Ryu et al., 2018).

Fingerprint types:
  - Morgan (ECFP4): 2048-bit circular fingerprint, captures chemical neighborhoods
  - MACCS keys: 166-bit, each bit has a known chemical meaning (more interpretable)

For a drug PAIR, we combine the two fingerprints into a single feature vector
using: concatenation, difference, or Hadamard product.

Usage:
    from src.features.molecular_features import smiles_to_morgan, build_pair_features
"""

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from src.utils.logger import get_logger
from src.utils.config import get_config

log = get_logger(__name__)
feat_cfg = get_config("feature")


def smiles_to_morgan(smiles: str,
                     radius: int = None,
                     n_bits: int = None) -> np.ndarray:
    """
    Converts a SMILES string to a Morgan (ECFP4) fingerprint.

    Args:
        smiles: molecular SMILES string
        radius: circular neighborhood radius (default from config)
        n_bits: fingerprint length (default from config)

    Returns:
        np.ndarray of shape (n_bits,) — binary vector of 0s and 1s
    """
    radius = radius or feat_cfg["molecular"]["morgan_radius"]
    n_bits = n_bits or feat_cfg["molecular"]["morgan_bits"]

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            log.warning(f"Invalid SMILES, returning zero vector: {smiles[:30]}")
            return np.zeros(n_bits, dtype=np.int8)

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception as e:
        log.error(f"Morgan fingerprint error: {e}")
        return np.zeros(n_bits, dtype=np.int8)


def smiles_to_maccs(smiles: str) -> np.ndarray:
    """
    Converts SMILES to MACCS keys fingerprint (167 bits).
    More interpretable than Morgan — each bit represents a specific chemical feature.
    Used in the explainability module to identify which substructures cause interactions.

    Returns:
        np.ndarray of shape (167,)
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(167, dtype=np.int8)
        fp = MACCSkeys.GenMACCSKeys(mol)
        arr = np.zeros(167, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception as e:
        log.error(f"MACCS fingerprint error: {e}")
        return np.zeros(167, dtype=np.int8)


def build_pair_features(smiles_a: str, smiles_b: str,
                         method: str = None) -> np.ndarray:
    """
    Creates a combined feature vector for a DRUG PAIR.
    This is the input to our baseline ML models.

    Combination methods:
      - "concat":   [fp_a | fp_b]       → 4096 dims  (default, keeps all info)
      - "diff":     |fp_a - fp_b|       → 2048 dims  (captures structural difference)
      - "hadamard": fp_a * fp_b         → 2048 dims  (captures shared substructures)

    Args:
        smiles_a, smiles_b: SMILES of each drug
        method: combination method (default from config)

    Returns:
        np.ndarray feature vector
    """
    method = method or feat_cfg["molecular"]["pair_combination_method"]
    fp_a = smiles_to_morgan(smiles_a)
    fp_b = smiles_to_morgan(smiles_b)

    if method == "concat":
        return np.concatenate([fp_a, fp_b])
    elif method == "diff":
        return np.abs(fp_a.astype(np.int16) - fp_b.astype(np.int16))
    elif method == "hadamard":
        return fp_a * fp_b
    else:
        raise ValueError(f"Unknown method '{method}'. Use: concat, diff, hadamard")


def build_feature_matrix(df_pairs: pd.DataFrame,
                          method: str = None) -> tuple:
    """
    Builds the full feature matrix X and label vector y from a pairs DataFrame.

    Args:
        df_pairs: DataFrame with columns smiles_a, smiles_b, label
        method: fingerprint combination method

    Returns:
        X: np.ndarray (n_pairs, n_features)
        y: np.ndarray (n_pairs,)
        feature_names: list of feature name strings
    """
    method = method or feat_cfg["molecular"]["pair_combination_method"]
    log.info(f"Building feature matrix for {len(df_pairs)} pairs using '{method}'")

    X, y = [], []
    for i, row in df_pairs.iterrows():
        feats = build_pair_features(row["smiles_a"], row["smiles_b"], method)
        X.append(feats)
        y.append(row["label"])

    X = np.array(X)
    y = np.array(y)

    n_bits = feat_cfg["molecular"]["morgan_bits"]
    if method == "concat":
        feature_names = ([f"drug_a_bit_{i}" for i in range(n_bits)] +
                         [f"drug_b_bit_{i}" for i in range(n_bits)])
    else:
        feature_names = [f"pair_bit_{i}" for i in range(X.shape[1])]

    log.info(f"Feature matrix shape: {X.shape} | Labels: {np.bincount(y)}")
    return X, y, feature_names


# ── Quick test ──────────────────────────────────────────────────
if __name__ == "__main__":
    smiles_aspirin  = "CC(=O)Oc1ccccc1C(=O)O"
    smiles_warfarin = "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"

    fp = smiles_to_morgan(smiles_aspirin)
    print(f"Morgan shape: {fp.shape}, active bits: {fp.sum()}")

    pair = build_pair_features(smiles_aspirin, smiles_warfarin, "concat")
    print(f"Pair feature shape: {pair.shape}")
