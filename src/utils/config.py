"""
src/utils/config.py
--------------------
Loads all YAML config files and provides them as Python dicts.
Also loads environment variables from .env

Usage:
    from src.utils.config import get_config, get_env
    
    data_cfg = get_config("data")
    model_cfg = get_config("model")
    api_key = get_env("DRUGBANK_API_KEY")
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Project root (2 levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def get_config(name: str) -> dict:
    """
    Load a YAML config file by name.

    Args:
        name: one of "data", "model", "feature", "app"

    Returns:
        dict with config values
    
    Example:
        cfg = get_config("data")
        drugs = cfg["starter_drugs"]
    """
    config_path = CONFIG_DIR / f"{name}_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_env(key: str, default: str = "") -> str:
    """
    Get an environment variable value.
    
    Args:
        key: variable name e.g. "DRUGBANK_API_KEY"
        default: fallback value if not set
    
    Returns:
        string value
    """
    return os.getenv(key, default)


def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    return PROJECT_ROOT


# ── Quick test ──────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = get_config("data")
    print("Starter drugs:", cfg["starter_drugs"][:5])
    print("Project root:", get_project_root())
