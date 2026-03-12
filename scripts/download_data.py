"""
scripts/download_data.py
-------------------------
Step 1: Download and save all drug data from APIs.

Run:
    python scripts/download_data.py
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))

from src.data.data_loader import load_all_data, build_pairs_dataset
from src.utils.logger import get_logger

log = get_logger(__name__)

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("STEP 1: Data Download Pipeline")
    log.info("=" * 60)
    data = load_all_data()
    df_pairs = build_pairs_dataset(
        data["chemical"],
        data["biological"],
        data["clinical"]
    )
    log.info(f"Done! {len(df_pairs)} drug pairs saved to data/processed/interaction_pairs.csv")
