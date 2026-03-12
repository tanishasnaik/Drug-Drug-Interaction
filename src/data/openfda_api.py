"""
src/data/openfda_api.py
------------------------
Member 1 — Data Engineering Lead

Fetches real-world adverse event data from the FDA's FAERS database via OpenFDA API.

WHY THIS MATTERS (from literature review):
- DeepDDI, MUFFIN, KGNN, DDIMDL — NONE of them used real-world clinical evidence
- DrugBank only uses static rules, not evidence from actual patient reports
- We use FAERS (FDA Adverse Event Reporting System) to score interactions
  based on how many real patients reported adverse events when taking drug pairs

This gives us the "clinical features" modality in our multi-modal fusion model.

Usage:
    from src.data.openfda_api import fetch_adverse_events, fetch_drug_pair_reports
"""

import time
import requests
import pandas as pd
from pathlib import Path
from src.utils.logger import get_logger
from src.utils.config import get_config, get_env

log = get_logger(__name__)
cfg = get_config("data")

OPENFDA_BASE_URL = cfg["openfda"]["base_url"]
API_KEY = get_env("OPENFDA_API_KEY", "")


def _build_params(search: str, limit: int) -> dict:
    params = {"search": search, "limit": limit}
    if API_KEY:
        params["api_key"] = API_KEY
    return params


def fetch_adverse_events(drug_name: str, limit: int = 100) -> list:
    """
    Fetches adverse event reports for a single drug.

    Args:
        drug_name: e.g. "warfarin"
        limit: max reports to fetch (max 1000 per call)

    Returns:
        list of FDA event report dicts
    """
    search = f'patient.drug.medicinalproduct:"{drug_name}"'
    params = _build_params(search, limit)

    try:
        resp = requests.get(OPENFDA_BASE_URL, params=params,
                            timeout=cfg["openfda"]["timeout"])
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        log.info(f"OpenFDA: {len(results)} adverse events for '{drug_name}'")
        return results
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 404:
            log.warning(f"No FDA adverse events found for: {drug_name}")
        else:
            log.error(f"OpenFDA HTTP error for '{drug_name}': {e}")
        return []
    except Exception as e:
        log.error(f"OpenFDA fetch failed for '{drug_name}': {e}")
        return []


def fetch_drug_pair_reports(drug_a: str, drug_b: str, limit: int = 100) -> dict:
    """
    Fetches adverse event reports where BOTH drugs appear together.
    This is the core signal for real-world interaction risk.

    Args:
        drug_a, drug_b: drug names
        limit: max reports

    Returns:
        dict with: drug_a, drug_b, report_count, reaction_counts (top reactions)
    """
    search = (
        f'patient.drug.medicinalproduct:"{drug_a}" AND '
        f'patient.drug.medicinalproduct:"{drug_b}"'
    )
    params = _build_params(search, limit)

    try:
        resp = requests.get(OPENFDA_BASE_URL, params=params,
                            timeout=cfg["openfda"]["timeout"])
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])

        # Count reaction types across reports
        reaction_counts = {}
        for report in results:
            for reaction in report.get("patient", {}).get("reaction", []):
                rxn = reaction.get("reactionmeddrapt", "unknown").lower()
                reaction_counts[rxn] = reaction_counts.get(rxn, 0) + 1

        # Sort reactions by frequency
        top_reactions = sorted(reaction_counts.items(),
                                key=lambda x: x[1], reverse=True)[:10]

        log.info(f"OpenFDA pair ({drug_a} + {drug_b}): {len(results)} co-reports")
        return {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "co_report_count": len(results),
            "top_reactions": dict(top_reactions),
        }
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 404:
            log.info(f"No co-reports found for pair: {drug_a} + {drug_b}")
        else:
            log.error(f"OpenFDA pair fetch error: {e}")
        return {"drug_a": drug_a, "drug_b": drug_b, "co_report_count": 0, "top_reactions": {}}
    except Exception as e:
        log.error(f"OpenFDA pair fetch failed: {e}")
        return {"drug_a": drug_a, "drug_b": drug_b, "co_report_count": 0, "top_reactions": {}}


def build_adverse_event_features(drug_list: list, save_path: str = None,
                                  delay: float = 1.0) -> pd.DataFrame:
    """
    Builds a report count feature for each drug (individual, not pairs).
    This becomes a feature in the clinical modality.

    Returns DataFrame: drug_name, report_count, top_reactions
    """
    records = []
    for i, drug in enumerate(drug_list):
        log.info(f"[{i+1}/{len(drug_list)}] FDA reports for: {drug}")
        events = fetch_adverse_events(drug, limit=100)
        records.append({
            "drug_name": drug,
            "fda_report_count": len(events),
        })
        time.sleep(delay)

    df = pd.DataFrame(records)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        log.info(f"Saved FDA features to {save_path}")

    return df


# ── Quick test ──────────────────────────────────────────────────
if __name__ == "__main__":
    result = fetch_drug_pair_reports("aspirin", "warfarin")
    print(result)
