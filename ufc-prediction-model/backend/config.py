"""Central configuration for paths, thresholds, and model parameters."""
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
CACHE_DIR = DATA_DIR / "cache"
MODELS_DIR = PROJECT_ROOT / "models" / "artifacts"
ALIASES_PATH = DATA_DIR / "fighter_aliases.json"

# Feature engineering
STYLE_THRESHOLD = 0.5
DEBUT_FIGHT_THRESHOLD = 3
FUZZY_MATCH_THRESHOLD = 85
FUZZY_REVIEW_THRESHOLD = 95

# Model
RANDOM_SEED = 42
TEST_YEAR_CUTOFF = 2025
CV_START_YEAR = 2020

# API
UFC_API_BASE = "https://ufcapi.aristotle.me"
UFC_API_DAILY_LIMIT = 100
FASTAPI_PORT = 8000
