"""
src/data/data_loader.py
------------------------
Member 1 — Data Engineering Lead

Orchestrates the full data ingestion pipeline:
  1. Fetch chemical data (PubChem)
  2. Fetch biological data (DrugBank)
  3. Fetch clinical data (OpenFDA)
  4. Build labeled drug pair dataset

Run this once to populate data/ before training.

Usage:
    python src/data/data_loader.py
    OR
    from src.data.data_loader import load_all_data, build_pairs_dataset
"""

import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path
from src.data.pubchem_api import fetch_compounds_batch
from src.data.drugbank_api import build_biological_features_batch, fetch_known_interactions
from src.data.openfda_api import build_adverse_event_features, fetch_drug_pair_reports
from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)
cfg = get_config("data")


def load_all_data(drug_list: list = None) -> dict:
    """
    Full data ingestion pipeline. Fetches from all three sources.

    Args:
        drug_list: list of drug names (defaults to config starter list)

    Returns:
        dict with keys: chemical, biological, clinical
    """
    if drug_list is None:
        drug_list = cfg["starter_drugs"]

    log.info(f"Starting data ingestion for {len(drug_list)} drugs")
    log.info("=" * 50)

    # ── Step 1: Chemical features from PubChem ──────────────────
    log.info("Step 1/3: Fetching chemical data from PubChem...")
    df_chemical = fetch_compounds_batch(
        drug_list,
        save_path="data/raw/pubchem/compounds.csv"
    )

    # ── Step 2: Biological features from DrugBank ───────────────
    log.info("Step 2/3: Fetching biological data from DrugBank...")
    df_biological = build_biological_features_batch(
        drug_list,
        save_path="data/raw/drugbank/enzyme_features.csv"
    )

    # ── Step 3: Clinical features from OpenFDA ──────────────────
    log.info("Step 3/3: Fetching clinical data from OpenFDA...")
    df_clinical = build_adverse_event_features(
        drug_list,
        save_path="data/raw/openfda/report_counts.csv"
    )

    log.info("Data ingestion complete!")
    return {
        "chemical":   df_chemical,
        "biological": df_biological,
        "clinical":   df_clinical,
    }


def build_pairs_dataset(df_chemical: pd.DataFrame,
                         df_biological: pd.DataFrame,
                         df_clinical: pd.DataFrame,
                         save_path: str = "data/processed/interaction_pairs.csv") -> pd.DataFrame:
    """
    Creates the drug pair dataset used for model training.

    For each pair of drugs:
      - Records their SMILES
      - Records enzyme overlap count (biological feature)
      - Records combined FDA report count (clinical feature)
      - Assigns a label (0=no interaction, 1=interaction)

    IMPORTANT: Labels here are synthetic (based on enzyme overlap + FDA reports).
    For real training, replace with ground truth from DrugBank DDI pairs.

    Returns:
        pd.DataFrame with columns: drug_a, drug_b, smiles_a, smiles_b,
                                   enzyme_overlap, fda_score, label
    """
    log.info("Building drug pair dataset...")

    # Merge all features into one lookup dict per drug
    chemical_map  = df_chemical.set_index("name").to_dict("index")
    bio_map        = df_biological.set_index("drug_name").to_dict("index")
    clinical_map   = df_clinical.set_index("drug_name").to_dict("index")

    valid_drugs = [d for d in df_chemical["name"].tolist()
                   if chemical_map.get(d, {}).get("smiles")]

    pairs = list(combinations(valid_drugs, 2))
    log.info(f"Building {len(pairs)} drug pairs from {len(valid_drugs)} valid drugs")

    rows = []
    for drug_a, drug_b in pairs:
        smiles_a = chemical_map.get(drug_a, {}).get("smiles", "")
        smiles_b = chemical_map.get(drug_b, {}).get("smiles", "")

        if not smiles_a or not smiles_b:
            continue

        # Enzyme overlap (biological modality)
        enzymes_a = set(bio_map.get(drug_a, {}).get("enzymes", "").split("|"))
        enzymes_b = set(bio_map.get(drug_b, {}).get("enzymes", "").split("|"))
        enzymes_a.discard("")
        enzymes_b.discard("")
        enzyme_overlap = len(enzymes_a & enzymes_b)

        # FDA report counts (clinical modality)
        fda_a = clinical_map.get(drug_a, {}).get("fda_report_count", 0)
        fda_b = clinical_map.get(drug_b, {}).get("fda_report_count", 0)
        fda_score = (fda_a + fda_b) / 2

        # Synthetic label: enzyme overlap > 0 OR high FDA count → interaction
        # TODO: Replace with DrugBank ground truth labels for real training
        import random
        random.seed(hash(drug_a + drug_b) % 1000)
        label = 1 if enzyme_overlap > 0 else random.randint(0, 1)

        rows.append({
            "drug_a":         drug_a,
            "drug_b":         drug_b,
            "smiles_a":       smiles_a,
            "smiles_b":       smiles_b,
            "enzyme_overlap": enzyme_overlap,
            "fda_score":      round(fda_score, 2),
            "label":          label,
        })

    df_pairs = pd.DataFrame(rows)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df_pairs.to_csv(save_path, index=False)
    log.info(f"Saved {len(df_pairs)} pairs to {save_path}")
    log.info(f"Label distribution: {df_pairs['label'].value_counts().to_dict()}")
    return df_pairs


# ── Full pipeline entry point ────────────────────────────────────
if __name__ == "__main__":
    data = load_all_data()
    df_pairs = build_pairs_dataset(
        data["chemical"],
        data["biological"],
        data["clinical"]
    )
    print(df_pairs.head(10))
