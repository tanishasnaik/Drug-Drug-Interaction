"""
src/data/drugbank_api.py
-------------------------
Member 1 — Data Engineering Lead

Fetches biological data from DrugBank: enzyme targets, metabolic pathways,
mechanisms of action, and known DDI pairs.

WHY THIS MATTERS (from literature review):
- MUFFIN used target proteins + pathways → higher accuracy than DeepDDI (chemical only)
- KGNN built a knowledge graph of drugs + targets + enzymes
- We use this as our "biological features" modality

DrugBank API requires a free account at: https://go.drugbank.com/
Set DRUGBANK_API_KEY in your .env file.

NOTE: If you don't have a DrugBank key yet, the module gracefully falls back
to a minimal hardcoded CYP enzyme dataset so other modules still work.

Usage:
    from src.data.drugbank_api import fetch_drug_targets, fetch_known_interactions
"""

import requests
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config import get_config, get_env

log = get_logger(__name__)
cfg = get_config("data")
API_KEY = get_env("DRUGBANK_API_KEY", "")
BASE_URL = cfg["drugbank"]["base_url"]

# ── CYP enzyme inhibition fallback data ─────────────────────────
# Source: well-known pharmacokinetic interactions (publicly documented)
# Used when DrugBank API is unavailable.
CYP_INHIBITION_FALLBACK = {
    "warfarin":      ["CYP2C9", "CYP3A4"],
    "fluoxetine":    ["CYP2D6", "CYP2C9"],
    "omeprazole":    ["CYP2C19"],
    "amiodarone":    ["CYP2D6", "CYP2C9", "CYP3A4"],
    "clarithromycin":["CYP3A4"],
    "itraconazole":  ["CYP3A4"],
    "ciprofloxacin": ["CYP1A2"],
    "metformin":     [],
    "aspirin":       [],
    "lisinopril":    [],
    "amlodipine":    ["CYP3A4"],
    "simvastatin":   ["CYP3A4"],
    "atorvastatin":  ["CYP3A4"],
    "clopidogrel":   ["CYP2C19"],
    "sertraline":    ["CYP2D6"],
    "gabapentin":    [],
    "ibuprofen":     ["CYP2C9"],
    "metoprolol":    ["CYP2D6"],
    "losartan":      ["CYP2C9"],
    "prednisone":    ["CYP3A4"],
    "furosemide":    [],
    "digoxin":       [],
}


def fetch_drug_targets(drug_name: str) -> dict:
    """
    Fetches enzyme/target data for a drug from DrugBank API.
    Falls back to CYP_INHIBITION_FALLBACK if no API key.

    Returns:
        dict with: drug_name, enzymes (list), targets (list)
    """
    if not API_KEY:
        log.warning("No DrugBank API key — using fallback CYP enzyme data")
        enzymes = CYP_INHIBITION_FALLBACK.get(drug_name.lower(), [])
        return {"drug_name": drug_name, "enzymes": enzymes, "targets": []}

    headers = {"Authorization": f"Token {API_KEY}"}
    try:
        url = f"{BASE_URL}/drugs/{drug_name}/enzymes"
        resp = requests.get(url, headers=headers,
                            timeout=cfg["drugbank"]["timeout"])
        resp.raise_for_status()
        data = resp.json()
        enzymes = [e.get("name", "") for e in data.get("enzymes", [])]
        targets = [t.get("name", "") for t in data.get("targets", [])]
        log.info(f"DrugBank: {drug_name} → {len(enzymes)} enzymes, {len(targets)} targets")
        return {"drug_name": drug_name, "enzymes": enzymes, "targets": targets}
    except Exception as e:
        log.error(f"DrugBank fetch failed for '{drug_name}': {e}")
        enzymes = CYP_INHIBITION_FALLBACK.get(drug_name.lower(), [])
        return {"drug_name": drug_name, "enzymes": enzymes, "targets": []}


def fetch_known_interactions(drug_name: str) -> list:
    """
    Fetches known DDI pairs from DrugBank for a drug.
    Used to build labeled training data (positive examples).

    Returns:
        list of dicts: {drug_a, drug_b, severity, description}
    """
    if not API_KEY:
        log.warning("No DrugBank API key — cannot fetch known interactions")
        return []

    headers = {"Authorization": f"Token {API_KEY}"}
    try:
        url = f"{BASE_URL}/drugs/{drug_name}/interactions"
        resp = requests.get(url, headers=headers,
                            timeout=cfg["drugbank"]["timeout"])
        resp.raise_for_status()
        data = resp.json()
        interactions = []
        for item in data.get("interactions", []):
            interactions.append({
                "drug_a": drug_name,
                "drug_b": item.get("drug_name", ""),
                "severity": item.get("severity", "unknown"),
                "description": item.get("description", ""),
            })
        log.info(f"DrugBank: {len(interactions)} known interactions for {drug_name}")
        return interactions
    except Exception as e:
        log.error(f"DrugBank interaction fetch failed for '{drug_name}': {e}")
        return []


def build_enzyme_overlap_feature(drug_a: str, drug_b: str) -> dict:
    """
    KEY INNOVATION: Detects shared enzyme metabolism between two drugs.

    If two drugs are both metabolized by CYP2C9 (for example), they compete
    for the same enzyme → one drug's levels can spike dangerously.

    This is the mechanistic explanation our model provides
    (e.g., "Both drugs inhibit CYP2C9 → overdose risk")

    Returns:
        dict: shared_enzymes, overlap_count, mechanism_hint
    """
    targets_a = fetch_drug_targets(drug_a)
    targets_b = fetch_drug_targets(drug_b)

    enzymes_a = set(targets_a.get("enzymes", []))
    enzymes_b = set(targets_b.get("enzymes", []))
    shared = list(enzymes_a & enzymes_b)

    mechanism = ""
    if shared:
        mechanism = (f"Both drugs are metabolized by {', '.join(shared)}. "
                     f"Co-administration may lead to competitive inhibition "
                     f"and elevated plasma levels of one or both drugs.")

    return {
        "drug_a": drug_a,
        "drug_b": drug_b,
        "shared_enzymes": shared,
        "enzyme_overlap_count": len(shared),
        "mechanism_hint": mechanism,
    }


def build_biological_features_batch(drug_list: list,
                                     save_path: str = None) -> pd.DataFrame:
    """
    Builds biological feature DataFrame for all drugs.
    Columns: drug_name, enzymes, enzyme_count
    """
    records = []
    for drug in drug_list:
        data = fetch_drug_targets(drug)
        records.append({
            "drug_name":    data["drug_name"],
            "enzymes":      "|".join(data["enzymes"]),  # pipe-separated for CSV
            "enzyme_count": len(data["enzymes"]),
        })

    df = pd.DataFrame(records)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        log.info(f"Saved biological features to {save_path}")
    return df


# ── Quick test ──────────────────────────────────────────────────
if __name__ == "__main__":
    overlap = build_enzyme_overlap_feature("warfarin", "fluoxetine")
    print(overlap)
    # Expected: both metabolized by CYP2C9 → shared_enzymes: ['CYP2C9']
