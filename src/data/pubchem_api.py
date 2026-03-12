"""
src/data/pubchem_api.py
------------------------
Member 1 — Data Engineering Lead

Fetches chemical data from PubChem (free, no API key required).
PubChem is our source for:
  - SMILES strings (needed to generate fingerprints for DeepDDI-style features)
  - Molecular formula and weight
  - CID (PubChem compound ID, used as a unique drug identifier)

Paper reference: DeepDDI (Ryu et al., 2018) used SMILES as primary input.
We extend this by combining with biological + clinical data.

Usage:
    from src.data.pubchem_api import fetch_compound, fetch_compounds_batch
"""

import time
import requests
import pubchempy as pcp
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config import get_config

log = get_logger(__name__)
cfg = get_config("data")


def fetch_compound(drug_name: str) -> dict:
    """
    Fetches chemical data for a single drug from PubChem.

    Args:
        drug_name: common drug name e.g. "aspirin", "warfarin"

    Returns:
        dict with: name, cid, smiles, molecular_formula, molecular_weight
        or empty dict if not found
    """
    log.info(f"PubChem: fetching '{drug_name}'")
    try:
        results = pcp.get_compounds(drug_name, "name")
        if not results:
            log.warning(f"No PubChem result for: {drug_name}")
            return {}

        c = results[0]
        data = {
            "name":              drug_name.lower().strip(),
            "cid":               c.cid,
            "smiles":            c.isomeric_smiles or c.canonical_smiles,
            "molecular_formula": c.molecular_formula,
            "molecular_weight":  c.molecular_weight,
            "iupac_name":        c.iupac_name,
        }
        log.info(f"  ✓ {drug_name} → CID {c.cid}")
        return data

    except Exception as e:
        log.error(f"PubChem fetch failed for '{drug_name}': {e}")
        return {}


def fetch_compounds_batch(drug_list: list, save_path: str = None,
                           delay: float = 0.5) -> pd.DataFrame:
    """
    Fetches PubChem data for a list of drugs and returns a DataFrame.

    Args:
        drug_list: list of drug name strings
        save_path: optional CSV save path e.g. "data/raw/pubchem/compounds.csv"
        delay: seconds to wait between API calls (be polite to the server!)

    Returns:
        pd.DataFrame with one row per drug
    """
    records = []
    failed = []

    for i, drug in enumerate(drug_list):
        log.info(f"[{i+1}/{len(drug_list)}] Fetching: {drug}")
        result = fetch_compound(drug)
        if result:
            records.append(result)
        else:
            failed.append(drug)
        time.sleep(delay)  # Rate limiting

    df = pd.DataFrame(records)

    if failed:
        log.warning(f"Failed to fetch {len(failed)} drugs: {failed}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        log.info(f"Saved {len(df)} compounds to {save_path}")

    log.info(f"Batch complete: {len(records)} success, {len(failed)} failed")
    return df


def load_starter_drugs() -> pd.DataFrame:
    """
    Convenience function: fetch all drugs listed in config/data_config.yaml
    and save to data/raw/pubchem/compounds.csv
    """
    drug_list = cfg["starter_drugs"]
    log.info(f"Loading {len(drug_list)} starter drugs from config")
    return fetch_compounds_batch(
        drug_list,
        save_path="data/raw/pubchem/compounds.csv"
    )


# ── Quick test ──────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_starter_drugs()
    print(df[["name", "cid", "molecular_weight"]].to_string())
