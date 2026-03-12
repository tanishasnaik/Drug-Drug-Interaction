"""
scripts/continuous_learning.py
--------------------------------
Monthly retraining pipeline.
Fetches new FDA adverse event data and retrains the model.

KEY INNOVATION vs prior work: All prior systems are static.
Our model retrains monthly with fresh real-world data from FDA.

Run manually OR schedule with a cron job:
    # Cron (first day of every month at 2am):
    # 0 2 1 * * /path/to/venv/bin/python /path/to/scripts/continuous_learning.py

Run manually:
    python scripts/continuous_learning.py
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))

import shutil
from datetime import datetime
import pandas as pd
from src.data.openfda_api import build_adverse_event_features
from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)
cfg = get_config("data")


def backup_current_model():
    """Backs up the current model before retraining."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    src = "models/baseline/xgboost.pkl"
    dst = f"models/baseline/xgboost_backup_{timestamp}.pkl"
    if os.path.exists(src):
        shutil.copy(src, dst)
        log.info(f"Model backed up to {dst}")


def update_fda_data():
    """Fetches latest adverse event data from FDA."""
    log.info("Fetching latest FDA adverse event data...")
    drug_list = cfg["starter_drugs"]
    df_new = build_adverse_event_features(
        drug_list,
        save_path="data/raw/openfda/report_counts.csv"
    )
    log.info(f"Updated FDA data: {len(df_new)} drugs")
    return df_new


def retrain_model():
    """Retrains the XGBoost model with updated data."""
    import subprocess
    log.info("Retraining model with updated data...")
    result = subprocess.run(
        ["python", "scripts/train_models.py"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        log.info("Model retrained successfully!")
    else:
        log.error(f"Retraining failed: {result.stderr}")


if __name__ == "__main__":
    log.info("=" * 60)
    log.info(f"Continuous Learning Pipeline — {datetime.now().strftime('%Y-%m-%d')}")
    log.info("=" * 60)

    backup_current_model()
    update_fda_data()
    retrain_model()

    log.info("✅ Continuous learning cycle complete!")
