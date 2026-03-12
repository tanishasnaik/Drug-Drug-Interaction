"""
src/inference/predictor.py
---------------------------
Member 4 — Frontend & Deployment Lead

Unified prediction engine that the Streamlit app calls.
Handles: data fetching → feature engineering → prediction → explanation → severity

This is the glue that connects all modules together.

Usage:
    from src.inference.predictor import DDIPredictor
    predictor = DDIPredictor()
    result = predictor.predict_pair("warfarin", "fluoxetine")
"""

import numpy as np
import json
from pathlib import Path
from src.data.pubchem_api import fetch_compound
from src.data.drugbank_api import build_enzyme_overlap_feature
from src.data.openfda_api import fetch_adverse_events
from src.features.molecular_features import build_pair_features
from src.features.biological_features import build_bio_feature_vector, get_mechanism_text
from src.features.clinical_features import build_clinical_vector
from src.explainability.shap_explainer import get_severity
from src.explainability.mechanism_extractor import MechanismExtractor
from src.utils.logger import get_logger

log = get_logger(__name__)


class DDIPredictor:
    """
    Main prediction class used by the Streamlit app.
    Orchestrates the full pipeline from drug name to interaction result.
    """

    def __init__(self, model_path: str = "models/baseline/xgboost.pkl",
                 feature_names_path: str = "models/metadata/feature_names.json"):
        """
        Loads the trained model and feature names.
        Falls back to demo mode if no model file exists yet.
        """
        self.model = None
        self.feature_names = None
        self.demo_mode = False
        self.mechanism_extractor = MechanismExtractor()

        # Load model
        try:
            import joblib
            self.model = joblib.load(model_path)
            log.info(f"Model loaded from {model_path}")
        except FileNotFoundError:
            log.warning(f"Model not found at {model_path} — running in DEMO mode")
            self.demo_mode = True

        # Load feature names
        try:
            with open(feature_names_path) as f:
                self.feature_names = json.load(f)
        except FileNotFoundError:
            log.warning("Feature names file not found — names will be auto-generated")

    def predict_pair(self, drug_a: str, drug_b: str) -> dict:
        """
        Full prediction pipeline for a pair of drugs.

        Args:
            drug_a, drug_b: drug names (e.g. "warfarin", "fluoxetine")

        Returns:
            dict with all prediction details, or dict with error key on failure
        """
        drug_a = drug_a.lower().strip()
        drug_b = drug_b.lower().strip()

        log.info(f"Predicting interaction: {drug_a} + {drug_b}")

        # ── Step 1: Fetch chemical data ──────────────────────────
        info_a = fetch_compound(drug_a)
        info_b = fetch_compound(drug_b)

        if not info_a or not info_a.get("smiles"):
            return {"error": f"Could not find chemical data for '{drug_a}'. Check spelling."}
        if not info_b or not info_b.get("smiles"):
            return {"error": f"Could not find chemical data for '{drug_b}'. Check spelling."}

        # ── Step 2: Build feature vector ─────────────────────────
        # Chemical modality
        X_mol = build_pair_features(info_a["smiles"], info_b["smiles"])

        # Biological modality
        bio_overlap = build_enzyme_overlap_feature(drug_a, drug_b)
        enzymes_a = bio_overlap.get("shared_enzymes", [])  # simplified
        enzymes_b = enzymes_a  # already shared
        X_bio = build_bio_feature_vector(
            bio_overlap.get("shared_enzymes", []),
            bio_overlap.get("shared_enzymes", [])
        )

        # Clinical modality
        fda_a = len(fetch_adverse_events(drug_a, limit=20))
        fda_b = len(fetch_adverse_events(drug_b, limit=20))
        X_clin = build_clinical_vector(fda_a, fda_b)

        # Fuse features
        X_fused = np.concatenate([X_mol, X_bio, X_clin])

        # ── Step 3: Predict ──────────────────────────────────────
        if self.demo_mode or self.model is None:
            # Demo: generate a heuristic probability
            enzyme_signal = bio_overlap.get("enzyme_overlap_count", 0) * 0.2
            fda_signal = min((fda_a + fda_b) / 200, 0.4)
            probability = min(0.1 + enzyme_signal + fda_signal, 0.95)
            log.warning("DEMO MODE: probability is heuristic, not from trained model")
        else:
            probability = float(
                self.model.predict_proba(X_fused.reshape(1, -1))[0][1]
            )

        # ── Step 4: Severity + Mechanism ─────────────────────────
        severity = get_severity(probability)

        from src.data.drugbank_api import fetch_drug_targets
        targets_a = fetch_drug_targets(drug_a)
        targets_b = fetch_drug_targets(drug_b)
        mechanism = self.mechanism_extractor.get_mechanism(
            drug_a, drug_b,
            targets_a.get("enzymes", []),
            targets_b.get("enzymes", []),
            probability
        )

        return {
            "drug_a":           info_a,
            "drug_b":           info_b,
            "probability":      round(probability, 4),
            "severity":         severity,
            "mechanism":        mechanism,
            "enzyme_overlap":   bio_overlap,
            "fda_reports_a":    fda_a,
            "fda_reports_b":    fda_b,
            "demo_mode":        self.demo_mode,
            "features":         X_fused,
        }

    def predict_multi(self, drug_list: list) -> list:
        """
        Predicts interactions for all pairs in a list of drugs.
        Supports polypharmacy (3+ drugs) — a key innovation over prior work.

        Args:
            drug_list: list of drug names (up to 10)

        Returns:
            list of prediction dicts, one per pair
        """
        from itertools import combinations
        pairs = list(combinations(drug_list, 2))
        log.info(f"Polypharmacy check: {len(drug_list)} drugs → {len(pairs)} pairs")

        results = []
        for drug_a, drug_b in pairs:
            result = self.predict_pair(drug_a, drug_b)
            results.append(result)
        return results


# ── Quick test ──────────────────────────────────────────────────
if __name__ == "__main__":
    predictor = DDIPredictor()
    result = predictor.predict_pair("warfarin", "aspirin")

    if "error" not in result:
        print(f"\nInteraction: {result['drug_a']['name']} + {result['drug_b']['name']}")
        print(f"Probability: {result['probability']:.1%}")
        print(f"Severity: {result['severity']['emoji']} {result['severity']['severity_label']}")
        print(f"Mechanism: {result['mechanism']['explanation']}")
    else:
        print("Error:", result["error"])
