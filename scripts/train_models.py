"""
scripts/train_models.py
-------------------------
Step 2: Train all baseline models using multi-modal features.

Run:
    python scripts/train_models.py
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))

import pandas as pd
from src.data.pubchem_api import fetch_compounds_batch
from src.features.feature_fusion import build_full_feature_matrix
from src.models.model_trainer import ModelTrainer
from src.utils.logger import get_logger

log = get_logger(__name__)

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("STEP 2: Model Training Pipeline")
    log.info("=" * 60)

    # Load processed pairs
    try:
        df_pairs = pd.read_csv("data/processed/interaction_pairs.csv")
        df_bio   = pd.read_csv("data/raw/drugbank/enzyme_features.csv")
        df_clin  = pd.read_csv("data/raw/openfda/report_counts.csv")
    except FileNotFoundError as e:
        log.error(f"Data file not found: {e}")
        log.error("Run scripts/download_data.py first!")
        exit(1)

    log.info(f"Loaded {len(df_pairs)} drug pairs")

    # Build multi-modal feature matrix
    X, y, feature_names = build_full_feature_matrix(df_pairs, df_bio, df_clin)

    # Train all models
    trainer = ModelTrainer()
    results = trainer.train_all(X, y, feature_names)

    log.info("\n✅ Training complete! Models saved to models/baseline/")
    log.info("✅ Metrics saved to results/metrics/baseline_metrics.json")
