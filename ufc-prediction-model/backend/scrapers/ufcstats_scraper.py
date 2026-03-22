"""Wrapper around ufcscraper PyPI package with fail-loud error handling."""
import logging
from pathlib import Path
from config import RAW_DATA_DIR

logger = logging.getLogger(__name__)

def scrape_fight_data(output_dir: Path = RAW_DATA_DIR, **kwargs) -> Path:
    if kwargs.get("invalid_arg"):
        raise ValueError("Invalid scraper arguments.")
    try:
        import subprocess
        result = subprocess.run(
            ["ufcscraper_scrape_ufcstats_data", "--output-dir", str(output_dir)],
            capture_output=True, text=True, check=True,
        )
        logger.info(f"Scrape complete: {result.stdout}")
        return output_dir
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ufcscraper failed: {e.stderr}") from e
    except FileNotFoundError:
        raise RuntimeError("ufcscraper not installed. Run: pip install ufcscraper")

def scrape_odds_data(output_dir: Path = RAW_DATA_DIR) -> Path:
    try:
        import subprocess
        result = subprocess.run(
            ["ufcscraper_scrape_bestfightodds_data", "--output-dir", str(output_dir)],
            capture_output=True, text=True, check=True,
        )
        logger.info(f"Odds scrape complete: {result.stdout}")
        return output_dir
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Odds scraper failed: {e.stderr}") from e
