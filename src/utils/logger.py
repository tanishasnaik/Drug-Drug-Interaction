"""
src/utils/logger.py
--------------------
Centralized logging for all modules.
Uses loguru for clean, colored, structured logs.

Usage:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("Fetching drug data...")
    log.warning("Missing SMILES for drug X")
    log.error("API call failed")
"""

import sys
from loguru import logger
from pathlib import Path

# Remove default loguru handler
logger.remove()

# Console handler — colored and readable
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# File handler — full debug logs saved to file
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logger.add(
    log_dir / "drug_interaction_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="1 day",      # New file every day
    retention="7 days",    # Keep 7 days of logs
    compression="zip",
)


def get_logger(name: str):
    """
    Returns a logger bound to the calling module's name.
    
    Usage:
        log = get_logger(__name__)
        log.info("Starting...")
    """
    return logger.bind(name=name)
