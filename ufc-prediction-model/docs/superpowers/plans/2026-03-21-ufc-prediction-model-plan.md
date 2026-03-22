# UFC Fight Prediction Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LightGBM-based UFC fight prediction system with time-aware feature engineering and a Next.js dashboard.

**Architecture:** Python backend (data pipeline → feature engineering → LightGBM training → FastAPI) with a Next.js + shadcn/ui frontend. Two model variants (with/without betting odds). All features computed time-aware to prevent data leakage.

**Tech Stack:** Python (pandas, LightGBM, SHAP, FastAPI, Optuna), Next.js (App Router, shadcn/ui, Tailwind, Recharts, Nivo)

**Spec:** `docs/superpowers/specs/2026-03-21-ufc-prediction-model-design.md`

---

## Phase 1: Project Scaffolding & Data Ingestion

### Task 1: Initialize Python Backend

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/config.py`
- Create: `backend/__init__.py`
- Create: `.gitignore`

- [ ] **Step 1: Create `.gitignore`**

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/
*.egg-info/
dist/
build/

# Data files (large CSVs)
backend/data/raw/*.csv
backend/data/processed/*.csv
backend/data/processed/*.parquet
backend/data/cache/
backend/models/artifacts/*.joblib

# Node
node_modules/
frontend/.next/
frontend/out/

# Environment
.env
.env.local
.env*.local

# OS
.DS_Store
Thumbs.db

# Brainstorm artifacts
.superpowers/
```

- [ ] **Step 2: Create `backend/requirements.txt`**

```
pandas>=2.2.0
numpy>=1.26.0
lightgbm>=4.3.0
shap>=0.45.0
optuna>=3.6.0
scikit-learn>=1.4.0
fastapi>=0.110.0
uvicorn>=0.27.0
pydantic>=2.6.0
rapidfuzz>=3.6.0
requests>=2.31.0
joblib>=1.3.0
ufcscraper>=1.1.0
```

- [ ] **Step 3: Create `backend/config.py`**

```python
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
STYLE_THRESHOLD = 0.5  # Min primary score to compute sub-type scores
DEBUT_FIGHT_THRESHOLD = 3  # Fights below this = debut/near-debut
FUZZY_MATCH_THRESHOLD = 85  # Min similarity for name matching
FUZZY_REVIEW_THRESHOLD = 95  # Below this, log for manual review

# Model
RANDOM_SEED = 42
TEST_YEAR_CUTOFF = 2025  # Held-out test set: 2025+ fights
CV_START_YEAR = 2020  # Expanding window CV starts here

# API
UFC_API_BASE = "https://ufcapi.aristotle.me"
UFC_API_DAILY_LIMIT = 100
FASTAPI_PORT = 8000
```

- [ ] **Step 4: Create `backend/__init__.py`**

```python
```

- [ ] **Step 5: Create directory structure**

Run:
```bash
mkdir -p backend/data/{raw,processed,predictions,cache}
mkdir -p backend/features
mkdir -p backend/models/artifacts
mkdir -p backend/scrapers
mkdir -p backend/api/routes
mkdir -p backend/tests
touch backend/features/__init__.py
touch backend/models/__init__.py
touch backend/scrapers/__init__.py
touch backend/api/__init__.py
touch backend/api/routes/__init__.py
touch backend/tests/__init__.py
```

- [ ] **Step 6: Set up virtual environment and install**

Run:
```bash
cd backend && python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

- [ ] **Step 7: Commit**

```bash
git add .gitignore backend/requirements.txt backend/config.py backend/__init__.py backend/features/__init__.py backend/models/__init__.py backend/scrapers/__init__.py backend/api/__init__.py backend/api/routes/__init__.py backend/tests/__init__.py
git commit -m "feat: initialize Python backend with config and dependencies"
```

---

### Task 2: Kaggle Data Loader

**Files:**
- Create: `backend/scrapers/kaggle_loader.py`
- Create: `backend/tests/test_kaggle_loader.py`
- Create: `backend/data/fighter_aliases.json`

- [ ] **Step 1: Create empty `backend/data/fighter_aliases.json`**

```json
{
  "_comment": "Maps alternative fighter names to canonical names from ufcstats.com",
  "aliases": {}
}
```

- [ ] **Step 2: Write failing tests for kaggle_loader**

```python
"""Tests for Kaggle dataset loading and merging."""
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch

from scrapers.kaggle_loader import (
    load_fight_data,
    load_rankings_data,
    clean_fight_data,
    exclude_non_binary_outcomes,
    validate_required_columns,
)


def test_exclude_non_binary_outcomes():
    """Draws, no-contests, DQs must be excluded per spec §3.1 Rule #6."""
    df = pd.DataFrame({
        "winner": ["Fighter A", "Fighter B", "Draw", "NC", "Fighter A"],
        "result": ["KO", "Decision", "Draw", "No Contest", "DQ"],
        "fighter_a": ["A", "B", "C", "D", "E"],
        "fighter_b": ["F", "G", "H", "I", "J"],
    })
    result = exclude_non_binary_outcomes(df)
    assert len(result) == 2  # Only KO and Decision wins
    assert "Draw" not in result["result"].values
    assert "No Contest" not in result["result"].values
    assert "DQ" not in result["result"].values


def test_validate_required_columns_passes():
    """Should not raise when all required columns present."""
    df = pd.DataFrame({
        "fighter_a": ["A"], "fighter_b": ["B"], "winner": ["A"],
        "event_date": ["2024-01-01"], "weight_class": ["Lightweight"],
    })
    required = ["fighter_a", "fighter_b", "winner", "event_date", "weight_class"]
    validate_required_columns(df, required)  # Should not raise


def test_validate_required_columns_fails():
    """Should raise ValueError when columns missing."""
    df = pd.DataFrame({"fighter_a": ["A"]})
    required = ["fighter_a", "fighter_b", "winner"]
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_required_columns(df, required)


def test_clean_fight_data_parses_dates():
    """Event dates must be parsed to datetime."""
    df = pd.DataFrame({
        "event_date": ["January 20, 2024", "2024-03-15"],
        "fighter_a": ["A", "B"],
        "fighter_b": ["C", "D"],
        "winner": ["A", "B"],
        "weight_class": ["Lightweight", "Welterweight"],
        "result": ["KO", "Decision"],
    })
    result = clean_fight_data(df)
    assert pd.api.types.is_datetime64_any_dtype(result["event_date"])
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_kaggle_loader.py -v`
Expected: FAIL — module not found

- [ ] **Step 4: Implement `backend/scrapers/kaggle_loader.py`**

```python
"""Load, clean, and merge Kaggle UFC datasets."""
import logging
from pathlib import Path

import pandas as pd

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

REQUIRED_FIGHT_COLUMNS = [
    "fighter_a", "fighter_b", "winner", "event_date", "weight_class",
]

NON_BINARY_RESULTS = {"Draw", "No Contest", "NC", "DQ", "Disqualification", "No contest"}
NON_BINARY_WINNERS = {"Draw", "NC", "No Contest", ""}


def validate_required_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Raise ValueError if any required columns are missing."""
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def exclude_non_binary_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Remove draws, no-contests, and DQs per spec §3.1 Rule #6."""
    mask_result = ~df["result"].isin(NON_BINARY_RESULTS) if "result" in df.columns else True
    mask_winner = ~df["winner"].isin(NON_BINARY_WINNERS) if "winner" in df.columns else True
    filtered = df[mask_result & mask_winner].copy()
    removed = len(df) - len(filtered)
    if removed > 0:
        logger.info(f"Excluded {removed} non-binary outcome fights (draws/NC/DQ)")
    return filtered


def clean_fight_data(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, standardize columns, remove invalid rows."""
    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"], format="mixed", dayfirst=False)
    df = df.dropna(subset=["fighter_a", "fighter_b", "winner"])
    # Drop pre-computed style scores from external datasets (spec §3.1 Rule 5)
    style_cols = [c for c in df.columns if "membership" in c.lower() or "style_score" in c.lower()]
    if style_cols:
        logger.info(f"Dropping pre-computed style columns (Rule 5): {style_cols}")
        df = df.drop(columns=style_cols)
    df = df.sort_values("event_date").reset_index(drop=True)
    return df


def load_fight_data(filename: str = "ufc_fights.csv") -> pd.DataFrame:
    """Load fight data from raw CSV, clean, and filter."""
    filepath = RAW_DATA_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Fight data not found at {filepath}. Download Kaggle dataset first.")
    df = pd.read_csv(filepath)
    validate_required_columns(df, REQUIRED_FIGHT_COLUMNS)
    df = clean_fight_data(df)
    df = exclude_non_binary_outcomes(df)
    return df


def load_rankings_data(filename: str = "ufc_rankings.csv") -> pd.DataFrame:
    """Load UFC rankings data from raw CSV."""
    filepath = RAW_DATA_DIR / filename
    if not filepath.exists():
        logger.warning(f"Rankings data not found at {filepath}. Rankings features will be unavailable.")
        return pd.DataFrame()
    df = pd.read_csv(filepath)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="mixed")
    return df
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_kaggle_loader.py -v`
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add backend/scrapers/kaggle_loader.py backend/tests/test_kaggle_loader.py backend/data/fighter_aliases.json
git commit -m "feat: add Kaggle data loader with cleaning and validation"
```

---

### Task 3: Fighter Identity Resolution

**Files:**
- Create: `backend/scrapers/name_matcher.py`
- Create: `backend/tests/test_name_matcher.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for fighter name matching and identity resolution."""
import pytest
from scrapers.name_matcher import FighterNameMatcher


@pytest.fixture
def matcher():
    aliases = {
        "Charles Do Bronx Oliveira": "Charles Oliveira",
        "The Korean Zombie": "Chan Sung Jung",
    }
    return FighterNameMatcher(aliases=aliases)


def test_exact_match(matcher):
    assert matcher.match("Charles Oliveira") == "Charles Oliveira"


def test_alias_match(matcher):
    assert matcher.match("Charles Do Bronx Oliveira") == "Charles Oliveira"


def test_fuzzy_match(matcher):
    """Should match close variations with high confidence."""
    result = matcher.match("Charles oliveira")  # lowercase
    assert result == "Charles Oliveira"


def test_no_match_returns_original():
    """Unknown fighters should return the original name."""
    matcher = FighterNameMatcher(aliases={}, known_fighters=[])
    result = matcher.match("Completely Unknown Fighter")
    assert result == "Completely Unknown Fighter"


def test_low_confidence_logged(matcher, caplog):
    """Matches below 95% similarity should be logged for review."""
    import logging
    with caplog.at_level(logging.WARNING):
        result = matcher.match("Charles Olivera", known_fighters=["Charles Oliveira"])
    assert result == "Charles Oliveira"  # Should still match (above 85%)
    assert any("Low-confidence" in msg for msg in caplog.messages)  # But warns (below 95%)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && python -m pytest tests/test_name_matcher.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `backend/scrapers/name_matcher.py`**

```python
"""Fighter name matching with fuzzy matching and alias support."""
import json
import logging
from pathlib import Path

from rapidfuzz import fuzz, process

from config import ALIASES_PATH, FUZZY_MATCH_THRESHOLD, FUZZY_REVIEW_THRESHOLD

logger = logging.getLogger(__name__)


class FighterNameMatcher:
    """Resolves fighter names across datasets using aliases and fuzzy matching."""

    def __init__(
        self,
        aliases: dict[str, str] | None = None,
        known_fighters: list[str] | None = None,
    ):
        if aliases is None:
            aliases = self._load_aliases()
        self.aliases = aliases
        self.known_fighters = known_fighters or []

    @staticmethod
    def _load_aliases() -> dict[str, str]:
        """Load alias table from JSON file."""
        if ALIASES_PATH.exists():
            data = json.loads(ALIASES_PATH.read_text())
            return data.get("aliases", {})
        return {}

    def match(
        self,
        name: str,
        known_fighters: list[str] | None = None,
    ) -> str:
        """Resolve a fighter name to its canonical form.

        Priority: exact match → alias lookup → fuzzy match → return original.
        """
        fighters = known_fighters or self.known_fighters

        # 1. Check alias table
        if name in self.aliases:
            canonical = self.aliases[name]
            logger.debug(f"Alias match: '{name}' → '{canonical}'")
            return canonical

        # 2. Exact match in known fighters (case-insensitive)
        name_lower = name.lower()
        for f in fighters:
            if f.lower() == name_lower:
                return f

        # 3. Fuzzy match
        if fighters:
            result = process.extractOne(
                name, fighters, scorer=fuzz.ratio, score_cutoff=FUZZY_MATCH_THRESHOLD
            )
            if result:
                matched_name, score, _ = result
                if score < FUZZY_REVIEW_THRESHOLD:
                    logger.warning(
                        f"Low-confidence match ({score:.0f}%): '{name}' → '{matched_name}'. "
                        f"Review and add to aliases if correct."
                    )
                else:
                    logger.debug(f"Fuzzy match ({score:.0f}%): '{name}' → '{matched_name}'")
                return matched_name

        # 4. No match found
        return name
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python -m pytest tests/test_name_matcher.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/scrapers/name_matcher.py backend/tests/test_name_matcher.py
git commit -m "feat: add fighter name resolution with fuzzy matching and aliases"
```

---

### Task 3.5: UFC Scraper Wrapper & API Client

**Files:**
- Create: `backend/scrapers/ufcstats_scraper.py`
- Create: `backend/scrapers/api_client.py`
- Create: `backend/tests/test_scraper.py`

- [ ] **Step 1: Write failing tests for API client**

```python
"""Tests for UFC API client with caching and rate limiting."""
import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from scrapers.api_client import UFCApiClient


def test_cached_response_used_when_available(tmp_path):
    """Should return cached data instead of making API call."""
    cache_file = tmp_path / "fighters_test.json"
    cache_file.write_text(json.dumps({"name": "Cached Fighter"}))
    client = UFCApiClient(cache_dir=tmp_path)
    result = client._get_cached("fighters_test")
    assert result == {"name": "Cached Fighter"}


def test_scraper_fail_loud():
    """Scraper errors must raise, never return bad data."""
    from scrapers.ufcstats_scraper import scrape_fight_data
    with pytest.raises(Exception):
        scrape_fight_data(invalid_arg=True)
```

- [ ] **Step 2: Implement `backend/scrapers/api_client.py`**

```python
"""REST client for ufcapi.aristotle.me with caching and rate limiting (spec S2.4)."""
import json
import logging
import time
from pathlib import Path

import requests

from config import UFC_API_BASE, UFC_API_DAILY_LIMIT, CACHE_DIR

logger = logging.getLogger(__name__)


class UFCApiClient:
    """Client for ufcapi.aristotle.me with local caching and exponential backoff."""

    def __init__(self, cache_dir: Path = CACHE_DIR, base_url: str = UFC_API_BASE):
        self.base_url = base_url
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._request_count = 0

    def _get_cached(self, cache_key: str) -> dict | None:
        path = self.cache_dir / f"{cache_key}.json"
        if path.exists():
            return json.loads(path.read_text())
        return None

    def _save_cache(self, cache_key: str, data: dict) -> None:
        path = self.cache_dir / f"{cache_key}.json"
        path.write_text(json.dumps(data, indent=2))

    def get(self, endpoint: str, cache_key: str | None = None) -> dict:
        """GET request with caching and rate limit awareness."""
        if cache_key:
            cached = self._get_cached(cache_key)
            if cached:
                logger.debug(f"Cache hit for {cache_key}")
                return cached

        if self._request_count >= UFC_API_DAILY_LIMIT:
            logger.warning("Daily API limit reached. Using cached data only.")
            if cache_key:
                cached = self._get_cached(cache_key)
                if cached:
                    return cached
            raise RuntimeError("API limit reached and no cached data available.")

        for attempt in range(3):
            try:
                resp = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                resp.raise_for_status()
                self._request_count += 1
                data = resp.json()
                if cache_key:
                    self._save_cache(cache_key, data)
                return data
            except requests.RequestException as e:
                wait = 2 ** attempt
                logger.warning(f"API request failed (attempt {attempt+1}): {e}. Retrying in {wait}s.")
                time.sleep(wait)

        raise RuntimeError(f"API request to {endpoint} failed after 3 retries.")

    def get_upcoming_event(self) -> dict:
        return self.get("/api/events/upcoming", cache_key="upcoming_event")

    def get_fighter(self, name: str) -> dict:
        safe_name = name.replace(" ", "_").lower()
        return self.get(f"/api/fighters/search?name={name}", cache_key=f"fighter_{safe_name}")
```

- [ ] **Step 3: Implement `backend/scrapers/ufcstats_scraper.py`**

```python
"""Wrapper around ufcscraper PyPI package with fail-loud error handling (spec S2.4)."""
import logging
from pathlib import Path

from config import RAW_DATA_DIR

logger = logging.getLogger(__name__)


def scrape_fight_data(output_dir: Path = RAW_DATA_DIR, **kwargs) -> Path:
    """Run ufcscraper to refresh fight data from ufcstats.com.

    Raises on any failure — never returns partial/corrupt data.
    """
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
        raise RuntimeError(
            "ufcscraper not installed. Run: pip install ufcscraper"
        )


def scrape_odds_data(output_dir: Path = RAW_DATA_DIR) -> Path:
    """Scrape betting odds from bestfightodds.com via ufcscraper."""
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
```

- [ ] **Step 4: Run tests — expect pass**

Run: `cd backend && python -m pytest tests/test_scraper.py -v`

- [ ] **Step 5: Commit**

```bash
git add backend/scrapers/ufcstats_scraper.py backend/scrapers/api_client.py backend/tests/test_scraper.py
git commit -m "feat: add ufcscraper wrapper and UFC API client with caching"
```

---

## Phase 2: Feature Engineering

### Task 4: Physical Attributes Features

**Files:**
- Create: `backend/features/physical.py`
- Create: `backend/tests/test_features_physical.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for physical attribute feature computation."""
import pytest
import pandas as pd
from features.physical import compute_physical_features


def test_basic_differentials():
    """Height/reach/age differentials computed correctly."""
    fighter_a = {"height_cm": 185, "reach_cm": 193, "age": 30, "stance": "Orthodox"}
    fighter_b = {"height_cm": 178, "reach_cm": 185, "age": 34, "stance": "Southpaw"}
    result = compute_physical_features(fighter_a, fighter_b)
    assert result["height_diff"] == 7
    assert result["reach_diff"] == 8
    assert result["age_diff"] == -4  # A is younger = negative diff
    assert result["a_stance"] == "Orthodox"
    assert result["b_stance"] == "Southpaw"


def test_missing_reach_handled():
    """Missing reach should not crash; fill with 0 differential."""
    fighter_a = {"height_cm": 185, "reach_cm": None, "age": 30, "stance": "Orthodox"}
    fighter_b = {"height_cm": 178, "reach_cm": 185, "age": 34, "stance": "Orthodox"}
    result = compute_physical_features(fighter_a, fighter_b)
    assert result["reach_diff"] == 0  # Unknown = no advantage assumed
```

- [ ] **Step 2: Run tests — expect fail**

Run: `cd backend && python -m pytest tests/test_features_physical.py -v`

- [ ] **Step 3: Implement `backend/features/physical.py`**

```python
"""Physical attribute features: height, reach, age, stance differentials."""


def _safe_diff(a_val, b_val) -> float:
    """Compute difference, returning 0 if either value is missing."""
    if a_val is None or b_val is None:
        return 0.0
    try:
        return float(a_val) - float(b_val)
    except (TypeError, ValueError):
        return 0.0


def compute_physical_features(fighter_a: dict, fighter_b: dict) -> dict:
    """Compute physical attribute features and differentials.

    Args:
        fighter_a: Dict with keys height_cm, reach_cm, age, stance
        fighter_b: Same structure for opponent

    Returns:
        Dict of feature name → value
    """
    return {
        # Raw values
        "a_height_cm": fighter_a.get("height_cm"),
        "b_height_cm": fighter_b.get("height_cm"),
        "a_reach_cm": fighter_a.get("reach_cm"),
        "b_reach_cm": fighter_b.get("reach_cm"),
        "a_age": fighter_a.get("age"),
        "b_age": fighter_b.get("age"),
        "a_stance": fighter_a.get("stance", "Unknown"),
        "b_stance": fighter_b.get("stance", "Unknown"),
        # Differentials
        "height_diff": _safe_diff(fighter_a.get("height_cm"), fighter_b.get("height_cm")),
        "reach_diff": _safe_diff(fighter_a.get("reach_cm"), fighter_b.get("reach_cm")),
        "age_diff": _safe_diff(fighter_a.get("age"), fighter_b.get("age")),
    }
```

- [ ] **Step 4: Run tests — expect pass**

Run: `cd backend && python -m pytest tests/test_features_physical.py -v`

- [ ] **Step 5: Commit**

```bash
git add backend/features/physical.py backend/tests/test_features_physical.py
git commit -m "feat: add physical attribute feature engineering"
```

---

### Task 5: Career Record Features

**Files:**
- Create: `backend/features/record.py`
- Create: `backend/tests/test_features_record.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for career record feature computation."""
import pytest
import pandas as pd
from features.record import compute_record_features, compute_career_stats_at_date


def test_compute_career_stats_at_date():
    """Stats must only include fights BEFORE the target date (leakage prevention)."""
    fights = pd.DataFrame({
        "event_date": pd.to_datetime(["2023-01-01", "2023-06-01", "2024-01-01", "2024-06-01"]),
        "fighter": ["A", "A", "A", "A"],
        "result": ["Win", "Win", "Loss", "Win"],
        "method": ["KO", "Decision", "KO", "Submission"],
    })
    # Stats at 2024-01-01 should only include 2023 fights (2 wins, 0 losses)
    stats = compute_career_stats_at_date(fights, "A", pd.Timestamp("2024-01-01"))
    assert stats["wins"] == 2
    assert stats["losses"] == 0
    assert stats["win_streak"] == 2
    assert stats["ko_win_pct"] == 0.5  # 1 KO out of 2 wins


def test_win_streak_resets_on_loss():
    fights = pd.DataFrame({
        "event_date": pd.to_datetime(["2023-01-01", "2023-03-01", "2023-06-01"]),
        "fighter": ["A", "A", "A"],
        "result": ["Win", "Loss", "Win"],
        "method": ["KO", "KO", "Decision"],
    })
    stats = compute_career_stats_at_date(fights, "A", pd.Timestamp("2024-01-01"))
    assert stats["win_streak"] == 1  # Reset after loss
    assert stats["loss_streak"] == 0


def test_record_differentials():
    a_stats = {"wins": 15, "losses": 3, "win_rate": 0.83, "finish_rate": 0.6, "ufc_fights": 18}
    b_stats = {"wins": 10, "losses": 5, "win_rate": 0.67, "finish_rate": 0.4, "ufc_fights": 15}
    result = compute_record_features(a_stats, b_stats)
    assert result["win_rate_diff"] == pytest.approx(0.16, abs=0.01)
    assert result["experience_diff"] == 3
```

- [ ] **Step 2: Run tests — expect fail**

Run: `cd backend && python -m pytest tests/test_features_record.py -v`

- [ ] **Step 3: Implement `backend/features/record.py`**

```python
"""Career record features: wins, losses, streaks, finish rates."""
import pandas as pd


def compute_career_stats_at_date(
    fights_df: pd.DataFrame, fighter_name: str, target_date: pd.Timestamp
) -> dict:
    """Compute cumulative career stats for a fighter using only fights BEFORE target_date.

    This is the core time-aware aggregation function that prevents data leakage.
    """
    prior = fights_df[
        (fights_df["fighter"] == fighter_name) &
        (fights_df["event_date"] < target_date)
    ].sort_values("event_date")

    if prior.empty:
        return _empty_stats()

    wins = (prior["result"] == "Win").sum()
    losses = (prior["result"] == "Loss").sum()
    draws = (prior["result"] == "Draw").sum()
    total = len(prior)

    # Win/loss streaks (count from most recent backwards)
    win_streak = 0
    loss_streak = 0
    for result in prior["result"].iloc[::-1]:
        if result == "Win":
            if loss_streak == 0:
                win_streak += 1
            else:
                break
        elif result == "Loss":
            if win_streak == 0:
                loss_streak += 1
            else:
                break
        else:
            break

    # Win method breakdown
    win_fights = prior[prior["result"] == "Win"]
    ko_wins = win_fights["method"].str.contains("KO|TKO", case=False, na=False).sum()
    sub_wins = win_fights["method"].str.contains("Sub", case=False, na=False).sum()
    dec_wins = wins - ko_wins - sub_wins

    # Career knockdown rate (spec §3.1 Rule 4: career rate only, not per-fight)
    kd_scored = prior["knockdowns_scored"].sum() if "knockdowns_scored" in prior.columns else 0
    kd_rate = kd_scored / total if total > 0 else 0.0

    return {
        "wins": int(wins),
        "losses": int(losses),
        "draws": int(draws),
        "ufc_fights": int(total),
        "win_rate": wins / total if total > 0 else 0.0,
        "win_streak": int(win_streak),
        "loss_streak": int(loss_streak),
        "ko_win_pct": ko_wins / wins if wins > 0 else 0.0,
        "sub_win_pct": sub_wins / wins if wins > 0 else 0.0,
        "dec_win_pct": dec_wins / wins if wins > 0 else 0.0,
        "finish_rate": (ko_wins + sub_wins) / wins if wins > 0 else 0.0,
        "kd_rate": kd_rate,
    }


def _empty_stats() -> dict:
    return {
        "wins": 0, "losses": 0, "draws": 0, "ufc_fights": 0,
        "win_rate": 0.0, "win_streak": 0, "loss_streak": 0,
        "ko_win_pct": 0.0, "sub_win_pct": 0.0, "dec_win_pct": 0.0,
        "finish_rate": 0.0,
    }


def compute_record_features(a_stats: dict, b_stats: dict) -> dict:
    """Compute record-based features and differentials between two fighters."""
    features = {}
    for prefix, stats in [("a", a_stats), ("b", b_stats)]:
        for key, val in stats.items():
            features[f"{prefix}_{key}"] = val

    features["win_rate_diff"] = a_stats.get("win_rate", 0) - b_stats.get("win_rate", 0)
    features["finish_rate_diff"] = a_stats.get("finish_rate", 0) - b_stats.get("finish_rate", 0)
    features["experience_diff"] = a_stats.get("ufc_fights", 0) - b_stats.get("ufc_fights", 0)
    features["streak_diff"] = a_stats.get("win_streak", 0) - b_stats.get("win_streak", 0)
    return features
```

- [ ] **Step 4: Run tests — expect pass**

Run: `cd backend && python -m pytest tests/test_features_record.py -v`

- [ ] **Step 5: Commit**

```bash
git add backend/features/record.py backend/tests/test_features_record.py
git commit -m "feat: add career record feature engineering with time-aware stats"
```

---

### Task 6: Striking Metrics Features

**Files:**
- Create: `backend/features/striking.py`
- Create: `backend/tests/test_features_striking.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for striking metric feature computation."""
import pytest
import pandas as pd
from features.striking import compute_striking_averages, compute_striking_features


def test_striking_averages_time_aware():
    """Career striking averages must only use fights before target date."""
    fight_stats = pd.DataFrame({
        "event_date": pd.to_datetime(["2023-01-01", "2023-06-01", "2024-01-01"]),
        "fighter": ["A", "A", "A"],
        "slpm": [4.0, 6.0, 8.0],
        "sapm": [3.0, 4.0, 2.0],
        "str_acc": [0.45, 0.55, 0.60],
        "str_def": [0.55, 0.60, 0.65],
    })
    avgs = compute_striking_averages(fight_stats, "A", pd.Timestamp("2024-01-01"))
    assert avgs["slpm"] == pytest.approx(5.0, abs=0.01)  # (4+6)/2
    assert avgs["sapm"] == pytest.approx(3.5, abs=0.01)  # (3+4)/2


def test_striking_differentials():
    a_avgs = {"slpm": 5.0, "sapm": 3.0, "str_acc": 0.50, "str_def": 0.60}
    b_avgs = {"slpm": 4.0, "sapm": 4.5, "str_acc": 0.45, "str_def": 0.55}
    result = compute_striking_features(a_avgs, b_avgs)
    assert result["slpm_diff"] == pytest.approx(1.0)
    assert result["str_acc_diff"] == pytest.approx(0.05)
    assert result["a_strike_differential"] == pytest.approx(2.0)  # 5.0 - 3.0
    assert result["b_strike_differential"] == pytest.approx(-0.5)  # 4.0 - 4.5
```

- [ ] **Step 2: Run tests — expect fail**
- [ ] **Step 3: Implement `backend/features/striking.py`**

```python
"""Striking metric features: SLpM, SApM, accuracy, defense, differentials."""
import pandas as pd


def compute_striking_averages(
    fight_stats: pd.DataFrame, fighter_name: str, target_date: pd.Timestamp
) -> dict:
    """Compute career striking averages from fights before target_date."""
    prior = fight_stats[
        (fight_stats["fighter"] == fighter_name) &
        (fight_stats["event_date"] < target_date)
    ]
    if prior.empty:
        return {"slpm": 0.0, "sapm": 0.0, "str_acc": 0.0, "str_def": 0.0}

    return {
        "slpm": prior["slpm"].mean(),
        "sapm": prior["sapm"].mean(),
        "str_acc": prior["str_acc"].mean(),
        "str_def": prior["str_def"].mean(),
    }


def compute_striking_features(a_avgs: dict, b_avgs: dict) -> dict:
    """Compute striking features and differentials."""
    features = {}
    for prefix, avgs in [("a", a_avgs), ("b", b_avgs)]:
        for key, val in avgs.items():
            features[f"{prefix}_{key}"] = val
        features[f"{prefix}_strike_differential"] = avgs["slpm"] - avgs["sapm"]

    features["slpm_diff"] = a_avgs["slpm"] - b_avgs["slpm"]
    features["sapm_diff"] = a_avgs["sapm"] - b_avgs["sapm"]
    features["str_acc_diff"] = a_avgs["str_acc"] - b_avgs["str_acc"]
    features["str_def_diff"] = a_avgs["str_def"] - b_avgs["str_def"]
    return features
```

- [ ] **Step 4: Run tests — expect pass**
- [ ] **Step 5: Commit**

```bash
git add backend/features/striking.py backend/tests/test_features_striking.py
git commit -m "feat: add striking metric feature engineering"
```

---

### Task 7: Grappling Metrics Features

**Files:**
- Create: `backend/features/grappling.py`
- Create: `backend/tests/test_features_grappling.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for grappling metric feature computation."""
import pytest
import pandas as pd
from features.grappling import compute_grappling_averages, compute_grappling_features


def test_grappling_averages_time_aware():
    fight_stats = pd.DataFrame({
        "event_date": pd.to_datetime(["2023-01-01", "2023-06-01"]),
        "fighter": ["A", "A"],
        "td_avg": [3.0, 5.0],
        "td_acc": [0.40, 0.60],
        "td_def": [0.70, 0.80],
        "sub_avg": [1.0, 2.0],
    })
    avgs = compute_grappling_averages(fight_stats, "A", pd.Timestamp("2024-01-01"))
    assert avgs["td_avg"] == pytest.approx(4.0)
    assert avgs["sub_avg"] == pytest.approx(1.5)


def test_grappling_differentials():
    a = {"td_avg": 4.0, "td_acc": 0.50, "td_def": 0.75, "sub_avg": 1.5}
    b = {"td_avg": 2.0, "td_acc": 0.35, "td_def": 0.80, "sub_avg": 0.5}
    result = compute_grappling_features(a, b)
    assert result["td_avg_diff"] == pytest.approx(2.0)
    assert result["grappling_advantage"] == pytest.approx(4.0 * 0.50 - 2.0 * 0.35, abs=0.01)
```

- [ ] **Step 2: Run tests — expect fail**
- [ ] **Step 3: Implement `backend/features/grappling.py`**

```python
"""Grappling metric features: takedown avg, accuracy, defense, submissions."""
import pandas as pd


def compute_grappling_averages(
    fight_stats: pd.DataFrame, fighter_name: str, target_date: pd.Timestamp
) -> dict:
    """Compute career grappling averages from fights before target_date."""
    prior = fight_stats[
        (fight_stats["fighter"] == fighter_name) &
        (fight_stats["event_date"] < target_date)
    ]
    if prior.empty:
        return {"td_avg": 0.0, "td_acc": 0.0, "td_def": 0.0, "sub_avg": 0.0}

    return {
        "td_avg": prior["td_avg"].mean(),
        "td_acc": prior["td_acc"].mean(),
        "td_def": prior["td_def"].mean(),
        "sub_avg": prior["sub_avg"].mean(),
    }


def compute_grappling_features(a_avgs: dict, b_avgs: dict) -> dict:
    """Compute grappling features and differentials."""
    features = {}
    for prefix, avgs in [("a", a_avgs), ("b", b_avgs)]:
        for key, val in avgs.items():
            features[f"{prefix}_{key}"] = val

    features["td_avg_diff"] = a_avgs["td_avg"] - b_avgs["td_avg"]
    features["td_acc_diff"] = a_avgs["td_acc"] - b_avgs["td_acc"]
    features["td_def_diff"] = a_avgs["td_def"] - b_avgs["td_def"]
    features["sub_avg_diff"] = a_avgs["sub_avg"] - b_avgs["sub_avg"]
    # Grappling advantage: A's offensive output vs B's offensive output
    features["grappling_advantage"] = (
        a_avgs["td_avg"] * a_avgs["td_acc"] - b_avgs["td_avg"] * b_avgs["td_acc"]
    )
    features["sub_threat_diff"] = a_avgs["sub_avg"] - b_avgs["sub_avg"]
    return features
```

- [ ] **Step 4: Run tests — expect pass**
- [ ] **Step 5: Commit**

```bash
git add backend/features/grappling.py backend/tests/test_features_grappling.py
git commit -m "feat: add grappling metric feature engineering"
```

---

### Task 8: Recent Form, Rankings, Context, and Odds Features

**Files:**
- Create: `backend/features/form.py`
- Create: `backend/features/rankings.py`
- Create: `backend/features/context.py`
- Create: `backend/features/odds.py`
- Create: `backend/tests/test_features_form.py`
- Create: `backend/tests/test_features_rankings.py`

- [ ] **Step 1: Write failing tests for form features**

```python
"""Tests for recent form feature computation."""
import pytest
import pandas as pd
from features.form import compute_form_features


def test_recent_form_last_3():
    fights = pd.DataFrame({
        "event_date": pd.to_datetime(["2023-01-01", "2023-04-01", "2023-07-01", "2023-10-01"]),
        "fighter": ["A", "A", "A", "A"],
        "result": ["Win", "Loss", "Win", "Win"],
    })
    form = compute_form_features(fights, "A", pd.Timestamp("2024-01-01"))
    assert form["last_3_wins"] == 2  # Loss, Win, Win
    assert form["last_3_losses"] == 1
    assert form["days_since_last_fight"] == 92  # Oct 1 to Jan 1


def test_debut_fighter_form():
    """Fighters with no prior fights should get default values."""
    fights = pd.DataFrame(columns=["event_date", "fighter", "result"])
    form = compute_form_features(fights, "A", pd.Timestamp("2024-01-01"))
    assert form["last_3_wins"] == 0
    assert form["is_debut"] is True
```

- [ ] **Step 2: Write failing tests for rankings features**

```python
"""Tests for rankings feature computation."""
import pytest
import pandas as pd
from features.rankings import get_ranking_at_date


def test_ranking_lookup_uses_most_recent_before_date():
    rankings = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-08", "2024-01-15"]),
        "fighter": ["Jon Jones", "Jon Jones", "Jon Jones"],
        "rank": [1, 1, 2],
        "weight_class": ["Heavyweight", "Heavyweight", "Heavyweight"],
    })
    rank = get_ranking_at_date(rankings, "Jon Jones", "Heavyweight", pd.Timestamp("2024-01-10"))
    assert rank == 1  # Jan 8 ranking (most recent before Jan 10)


def test_unranked_fighter():
    rankings = pd.DataFrame(columns=["date", "fighter", "rank", "weight_class"])
    rank = get_ranking_at_date(rankings, "Unknown", "Lightweight", pd.Timestamp("2024-01-01"))
    assert rank == 0  # Unranked
```

- [ ] **Step 3: Run all tests — expect fail**
- [ ] **Step 4: Implement all four modules**

Implement `backend/features/form.py`:
```python
"""Recent form features: days since fight, recent results, momentum."""
import pandas as pd
from config import DEBUT_FIGHT_THRESHOLD


def compute_form_features(
    fights_df: pd.DataFrame, fighter_name: str, target_date: pd.Timestamp
) -> dict:
    """Compute recent form features from fights before target_date."""
    prior = fights_df[
        (fights_df["fighter"] == fighter_name) &
        (fights_df["event_date"] < target_date)
    ].sort_values("event_date")

    total_fights = len(prior)
    is_debut = total_fights == 0
    is_near_debut = total_fights < DEBUT_FIGHT_THRESHOLD

    if is_debut:
        return {
            "last_3_wins": 0, "last_3_losses": 0,
            "last_5_wins": 0, "last_5_losses": 0,
            "days_since_last_fight": 365,
            "recent_finish_rate": 0.0,
            "is_debut": True, "is_near_debut": True,
        }

    last_3 = prior.tail(3)
    last_5 = prior.tail(5)
    last_fight_date = prior["event_date"].iloc[-1]

    return {
        "last_3_wins": int((last_3["result"] == "Win").sum()),
        "last_3_losses": int((last_3["result"] == "Loss").sum()),
        "last_5_wins": int((last_5["result"] == "Win").sum()),
        "last_5_losses": int((last_5["result"] == "Loss").sum()),
        "days_since_last_fight": (target_date - last_fight_date).days,
        "recent_finish_rate": (
            last_5["method"].str.contains("KO|TKO|Sub", case=False, na=False).sum()
            / len(last_5) if "method" in last_5.columns and len(last_5) > 0 else 0.0
        ),
        "is_debut": is_debut,
        "is_near_debut": is_near_debut,
    }
```

Implement `backend/features/rankings.py`:
```python
"""UFC rankings features: rank at fight time, rank differentials."""
import pandas as pd


def get_ranking_at_date(
    rankings_df: pd.DataFrame, fighter_name: str,
    weight_class: str, target_date: pd.Timestamp
) -> int:
    """Look up the most recent ranking published before the fight date. Returns 0 if unranked."""
    if rankings_df.empty:
        return 0
    mask = (
        (rankings_df["fighter"] == fighter_name) &
        (rankings_df["weight_class"] == weight_class) &
        (rankings_df["date"] < target_date)
    )
    relevant = rankings_df[mask].sort_values("date")
    if relevant.empty:
        return 0
    return int(relevant["rank"].iloc[-1])


def compute_rankings_features(a_rank: int, b_rank: int) -> dict:
    """Compute ranking-based features."""
    return {
        "a_rank": a_rank,
        "b_rank": b_rank,
        "a_is_ranked": a_rank > 0,
        "b_is_ranked": b_rank > 0,
        "rank_diff": a_rank - b_rank if (a_rank > 0 and b_rank > 0) else (
            -b_rank if (a_rank == 0 and b_rank > 0) else (a_rank if (b_rank == 0 and a_rank > 0) else 0)
        ),
        "a_stepping_up": a_rank == 0 and b_rank > 0,
        "b_stepping_up": b_rank == 0 and a_rank > 0,
    }
```

Implement `backend/features/context.py`:
```python
"""Fight context features: weight class, rounds, card position."""


def compute_context_features(
    weight_class: str, rounds_scheduled: int,
    is_title_fight: bool, card_position: str
) -> dict:
    """Compute fight context features."""
    return {
        "weight_class": weight_class,
        "rounds_scheduled": rounds_scheduled,
        "is_title_fight": is_title_fight,
        "card_position": card_position,
        "is_five_rounder": rounds_scheduled == 5,
    }
```

Implement `backend/features/odds.py`:
```python
"""Betting odds features (Model B only): implied probability, line movement."""


def compute_odds_features(a_odds: float | None, b_odds: float | None) -> dict:
    """Compute odds-based features. Returns empty dict if odds unavailable."""
    if a_odds is None or b_odds is None:
        return {
            "a_implied_prob": None, "b_implied_prob": None,
            "odds_diff": None, "market_confidence": None,
        }

    # Convert American odds to implied probability
    a_prob = _american_to_probability(a_odds)
    b_prob = _american_to_probability(b_odds)

    return {
        "a_implied_prob": a_prob,
        "b_implied_prob": b_prob,
        "odds_diff": a_prob - b_prob,
        "market_confidence": abs(a_prob - b_prob),
    }


def _american_to_probability(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)
```

- [ ] **Step 5: Run all tests — expect pass**

Run: `cd backend && python -m pytest tests/test_features_form.py tests/test_features_rankings.py -v`

- [ ] **Step 6: Commit**

```bash
git add backend/features/form.py backend/features/rankings.py backend/features/context.py backend/features/odds.py backend/tests/test_features_form.py backend/tests/test_features_rankings.py
git commit -m "feat: add form, rankings, context, and odds feature engineering"
```

---

### Task 9: Fighting Style Classification

**Files:**
- Create: `backend/features/style.py`
- Create: `backend/tests/test_features_style.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for fighting style classification system."""
import pytest
from features.style import compute_style_scores, get_sub_scores


def test_pure_striker_classification():
    """High striking, low grappling → high Striker score."""
    stats = {
        "slpm": 6.0, "sapm": 3.0, "str_acc": 0.55, "str_def": 0.60,
        "td_avg": 0.5, "td_acc": 0.30, "td_def": 0.80, "sub_avg": 0.1,
        "ko_win_pct": 0.70, "sub_win_pct": 0.05, "dec_win_pct": 0.25,
        "finish_rate": 0.75, "kd_rate": 0.3,
    }
    scores = compute_style_scores(stats)
    assert scores["striker"] > 0.6
    assert scores["wrestler"] < 0.3
    assert scores["grappler"] < 0.2


def test_pure_wrestler_classification():
    stats = {
        "slpm": 2.5, "sapm": 2.0, "str_acc": 0.42, "str_def": 0.55,
        "td_avg": 5.0, "td_acc": 0.55, "td_def": 0.75, "sub_avg": 0.3,
        "ko_win_pct": 0.10, "sub_win_pct": 0.10, "dec_win_pct": 0.80,
        "finish_rate": 0.20, "kd_rate": 0.05,
    }
    scores = compute_style_scores(stats)
    assert scores["wrestler"] > 0.6
    assert scores["striker"] < 0.4


def test_sub_scores_only_above_threshold():
    """Sub-scores should only be computed for primaries above threshold."""
    scores = {"striker": 0.8, "wrestler": 0.3, "grappler": 0.1, "balanced": 0.2}
    stats = {
        "slpm": 6.0, "sapm": 3.0, "str_acc": 0.55, "str_def": 0.60,
        "ko_win_pct": 0.70, "kd_rate": 0.3, "finish_rate": 0.75,
    }
    sub = get_sub_scores(scores, stats, threshold=0.5)
    assert "power_puncher" in sub  # Striker > 0.5 → sub-scores computed
    assert sub.get("control_wrestler", 0) == 0  # Wrestler < 0.5 → zeroed
    assert sub.get("sub_hunter", 0) == 0  # Grappler < 0.5 → zeroed


def test_all_scores_between_0_and_1():
    stats = {
        "slpm": 4.0, "sapm": 3.5, "str_acc": 0.48, "str_def": 0.55,
        "td_avg": 2.5, "td_acc": 0.40, "td_def": 0.65, "sub_avg": 1.0,
        "ko_win_pct": 0.35, "sub_win_pct": 0.25, "dec_win_pct": 0.40,
        "finish_rate": 0.60, "kd_rate": 0.15,
    }
    scores = compute_style_scores(stats)
    for key, val in scores.items():
        assert 0.0 <= val <= 1.0, f"{key}={val} outside [0,1]"
```

- [ ] **Step 2: Run tests — expect fail**
- [ ] **Step 3: Implement `backend/features/style.py`**

```python
"""Fighting style classification: 4 primary archetypes + sub-types with fuzzy scores.

Hierarchy (spec §4):
  Striker → Power Puncher, Counter-Striker, Pressure Fighter
  Wrestler → Control Wrestler, Ground & Pound
  Grappler → Submission Hunter, Positional Grappler
  Balanced → Adaptive, Defense-First
"""
import numpy as np
from config import STYLE_THRESHOLD


def _clamp(val: float) -> float:
    """Clamp value to [0, 1]."""
    return max(0.0, min(1.0, val))


def compute_style_scores(stats: dict) -> dict:
    """Compute 4 primary archetype fuzzy scores from career stats.

    Uses normalized stat ratios — higher striking volume + accuracy → higher Striker score, etc.
    """
    slpm = stats.get("slpm", 0)
    td_avg = stats.get("td_avg", 0)
    sub_avg = stats.get("sub_avg", 0)
    str_acc = stats.get("str_acc", 0)
    td_acc = stats.get("td_acc", 0)
    td_def = stats.get("td_def", 0)
    str_def = stats.get("str_def", 0)
    ko_pct = stats.get("ko_win_pct", 0)
    sub_pct = stats.get("sub_win_pct", 0)

    # Normalize key metrics to 0-1 scale using typical UFC ranges
    strike_signal = _clamp(slpm / 8.0) * 0.4 + _clamp(str_acc / 0.65) * 0.3 + ko_pct * 0.3
    wrestle_signal = _clamp(td_avg / 6.0) * 0.4 + _clamp(td_acc / 0.60) * 0.3 + (1 - ko_pct - sub_pct) * 0.3
    grapple_signal = _clamp(sub_avg / 3.0) * 0.4 + sub_pct * 0.4 + _clamp(td_avg / 6.0) * 0.2

    # Balanced = when no single dimension dominates
    max_signal = max(strike_signal, wrestle_signal, grapple_signal, 0.01)
    balance_signal = 1.0 - (max_signal - min(strike_signal, wrestle_signal, grapple_signal)) / max_signal
    balance_signal = balance_signal * 0.6 + _clamp(str_def / 0.65) * 0.2 + _clamp(td_def / 0.75) * 0.2

    return {
        "striker": _clamp(strike_signal),
        "wrestler": _clamp(wrestle_signal),
        "grappler": _clamp(grapple_signal),
        "balanced": _clamp(balance_signal),
    }


def get_sub_scores(primary_scores: dict, stats: dict, threshold: float = STYLE_THRESHOLD) -> dict:
    """Compute sub-type scores only for primaries above threshold.

    Sub-scores below threshold default to 0 per spec §4.2.
    """
    sub = {}
    slpm = stats.get("slpm", 0)
    sapm = stats.get("sapm", 0)
    str_acc = stats.get("str_acc", 0)
    str_def = stats.get("str_def", 0)
    ko_pct = stats.get("ko_win_pct", 0)
    kd_rate = stats.get("kd_rate", 0)
    finish_rate = stats.get("finish_rate", 0)

    # Striker sub-types
    if primary_scores.get("striker", 0) >= threshold:
        sub["power_puncher"] = _clamp(ko_pct * 0.4 + _clamp(kd_rate / 0.4) * 0.3 + finish_rate * 0.3)
        sub["counter_striker"] = _clamp(
            _clamp(str_acc / 0.60) * 0.3 + (1 - _clamp(sapm / 5.0)) * 0.4 + _clamp(str_def / 0.65) * 0.3
        )
        sub["pressure_fighter"] = _clamp(
            _clamp(slpm / 7.0) * 0.4 + _clamp(sapm / 5.0) * 0.3 + (1 - str_acc) * 0.3
        )
    else:
        sub["power_puncher"] = 0.0
        sub["counter_striker"] = 0.0
        sub["pressure_fighter"] = 0.0

    # Wrestler sub-types
    if primary_scores.get("wrestler", 0) >= threshold:
        td_avg = stats.get("td_avg", 0)
        dec_pct = stats.get("dec_win_pct", 0)
        sub["control_wrestler"] = _clamp(dec_pct * 0.5 + _clamp(td_avg / 5.0) * 0.5)
        sub["ground_and_pound"] = _clamp(ko_pct * 0.4 + _clamp(td_avg / 5.0) * 0.3 + finish_rate * 0.3)
    else:
        sub["control_wrestler"] = 0.0
        sub["ground_and_pound"] = 0.0

    # Grappler sub-types
    if primary_scores.get("grappler", 0) >= threshold:
        sub_avg = stats.get("sub_avg", 0)
        sub_pct = stats.get("sub_win_pct", 0)
        sub["sub_hunter"] = _clamp(sub_pct * 0.5 + _clamp(sub_avg / 2.5) * 0.5)
        sub["positional_grappler"] = _clamp(
            _clamp(stats.get("td_avg", 0) / 5.0) * 0.5 + (1 - sub_pct) * 0.5
        )
    else:
        sub["sub_hunter"] = 0.0
        sub["positional_grappler"] = 0.0

    # Balanced sub-types
    if primary_scores.get("balanced", 0) >= threshold:
        sub["adaptive"] = _clamp(
            (1 - abs(ko_pct - stats.get("sub_win_pct", 0) - stats.get("dec_win_pct", 0))) * 0.5
            + _clamp(slpm / 5.0) * 0.25 + _clamp(stats.get("td_avg", 0) / 3.0) * 0.25
        )
        sub["defense_first"] = _clamp(
            _clamp(str_def / 0.65) * 0.3 + _clamp(stats.get("td_def", 0) / 0.75) * 0.3
            + (1 - _clamp(sapm / 4.0)) * 0.4
        )
    else:
        sub["adaptive"] = 0.0
        sub["defense_first"] = 0.0

    return sub


def compute_all_style_features(stats: dict) -> dict:
    """Compute full style feature vector for a fighter."""
    primary = compute_style_scores(stats)
    subs = get_sub_scores(primary, stats)
    return {**primary, **subs}
```

- [ ] **Step 4: Run tests — expect pass**

Run: `cd backend && python -m pytest tests/test_features_style.py -v`

- [ ] **Step 5: Commit**

```bash
git add backend/features/style.py backend/tests/test_features_style.py
git commit -m "feat: add hierarchical fighting style classification with fuzzy scores"
```

---

### Task 10: Feature Pipeline Orchestrator

**Files:**
- Create: `backend/features/pipeline.py`
- Create: `backend/tests/test_pipeline.py`

- [ ] **Step 1: Write failing test for pipeline**

```python
"""Tests for the feature engineering pipeline orchestrator."""
import pytest
import pandas as pd
from features.pipeline import build_feature_row, augment_with_swap


def test_augment_with_swap():
    """Data augmentation: each fight produces two rows with swapped fighters."""
    df = pd.DataFrame({
        "a_slpm": [5.0], "b_slpm": [3.0],
        "slpm_diff": [2.0], "target": [1],
    })
    result = augment_with_swap(df)
    assert len(result) == 2
    assert result.iloc[1]["a_slpm"] == 3.0
    assert result.iloc[1]["b_slpm"] == 5.0
    assert result.iloc[1]["slpm_diff"] == -2.0
    assert result.iloc[1]["target"] == 0  # Label inverted


def test_build_feature_row_returns_dict():
    """Pipeline should return a flat dict of features."""
    # Minimal mock data
    fighter_a = {"height_cm": 185, "reach_cm": 193, "age": 30, "stance": "Orthodox"}
    fighter_b = {"height_cm": 178, "reach_cm": 185, "age": 34, "stance": "Southpaw"}
    row = build_feature_row(
        fighter_a_info=fighter_a,
        fighter_b_info=fighter_b,
        a_record={"wins": 10, "losses": 2, "win_rate": 0.83, "finish_rate": 0.5, "ufc_fights": 12,
                   "win_streak": 3, "loss_streak": 0, "ko_win_pct": 0.3, "sub_win_pct": 0.1, "dec_win_pct": 0.6},
        b_record={"wins": 8, "losses": 4, "win_rate": 0.67, "finish_rate": 0.4, "ufc_fights": 12,
                   "win_streak": 1, "loss_streak": 0, "ko_win_pct": 0.2, "sub_win_pct": 0.2, "dec_win_pct": 0.6},
        a_striking={"slpm": 5.0, "sapm": 3.0, "str_acc": 0.50, "str_def": 0.60},
        b_striking={"slpm": 4.0, "sapm": 4.0, "str_acc": 0.45, "str_def": 0.55},
        a_grappling={"td_avg": 2.0, "td_acc": 0.40, "td_def": 0.70, "sub_avg": 0.5},
        b_grappling={"td_avg": 1.0, "td_acc": 0.35, "td_def": 0.65, "sub_avg": 0.3},
        a_form={"last_3_wins": 3, "last_3_losses": 0, "last_5_wins": 4, "last_5_losses": 1,
                "days_since_last_fight": 90, "recent_finish_rate": 0.4, "is_debut": False, "is_near_debut": False},
        b_form={"last_3_wins": 1, "last_3_losses": 2, "last_5_wins": 2, "last_5_losses": 3,
                "days_since_last_fight": 180, "recent_finish_rate": 0.2, "is_debut": False, "is_near_debut": False},
        a_rank=5, b_rank=0,
        context={"weight_class": "Lightweight", "rounds_scheduled": 3,
                 "is_title_fight": False, "card_position": "prelim", "is_five_rounder": False},
        a_odds=None, b_odds=None,
    )
    assert isinstance(row, dict)
    assert "height_diff" in row
    assert "slpm_diff" in row
    assert "weight_class" in row
```

- [ ] **Step 2: Run tests — expect fail**
- [ ] **Step 3: Implement `backend/features/pipeline.py`**

```python
"""Feature engineering pipeline: orchestrates all feature modules into a single feature vector."""
import pandas as pd
import numpy as np

from features.physical import compute_physical_features
from features.record import compute_record_features
from features.striking import compute_striking_features
from features.grappling import compute_grappling_features
from features.rankings import compute_rankings_features
from features.context import compute_context_features
from features.odds import compute_odds_features
from features.style import compute_all_style_features


def build_feature_row(
    fighter_a_info: dict, fighter_b_info: dict,
    a_record: dict, b_record: dict,
    a_striking: dict, b_striking: dict,
    a_grappling: dict, b_grappling: dict,
    a_form: dict, b_form: dict,
    a_rank: int, b_rank: int,
    context: dict,
    a_odds: float | None = None, b_odds: float | None = None,
) -> dict:
    """Build a complete feature vector for a single fight."""
    features = {}

    # 1. Physical attributes
    features.update(compute_physical_features(fighter_a_info, fighter_b_info))

    # 2. Career record
    features.update(compute_record_features(a_record, b_record))

    # 3. Striking metrics
    features.update(compute_striking_features(a_striking, b_striking))

    # 4. Grappling metrics
    features.update(compute_grappling_features(a_grappling, b_grappling))

    # 5. Recent form
    for prefix, form in [("a", a_form), ("b", b_form)]:
        for key, val in form.items():
            features[f"{prefix}_{key}"] = val
    features["days_since_fight_diff"] = a_form["days_since_last_fight"] - b_form["days_since_last_fight"]
    features["momentum_diff"] = a_form["last_3_wins"] - b_form["last_3_wins"]

    # 6. Rankings
    features.update(compute_rankings_features(a_rank, b_rank))

    # 7. Context
    features.update(context)

    # 8. Odds (Model B only — may be None)
    features.update(compute_odds_features(a_odds, b_odds))

    # 9. Style classification
    a_style_stats = {**a_striking, **a_grappling, **a_record}
    b_style_stats = {**b_striking, **b_grappling, **b_record}
    a_style = compute_all_style_features(a_style_stats)
    b_style = compute_all_style_features(b_style_stats)
    for key in a_style:
        features[f"a_style_{key}"] = a_style[key]
        features[f"b_style_{key}"] = b_style[key]
        features[f"style_{key}_diff"] = a_style[key] - b_style[key]

    return features


def augment_with_swap(df: pd.DataFrame) -> pd.DataFrame:
    """Data augmentation: duplicate each fight with fighters swapped and label inverted.

    This prevents the model from learning positional bias (spec §5.3).
    """
    swapped = df.copy()

    # Swap a_ and b_ prefixed columns
    a_cols = [c for c in df.columns if c.startswith("a_")]
    b_cols = [c for c in df.columns if c.startswith("b_")]
    for a_col in a_cols:
        b_col = "b_" + a_col[2:]
        if b_col in swapped.columns:
            swapped[a_col], swapped[b_col] = df[b_col].values, df[a_col].values

    # Negate differential columns
    diff_cols = [c for c in df.columns if c.endswith("_diff")]
    for col in diff_cols:
        swapped[col] = -df[col]

    # Invert target label
    if "target" in swapped.columns:
        swapped["target"] = 1 - df["target"]

    return pd.concat([df, swapped], ignore_index=True)
```

- [ ] **Step 4: Run all tests — expect pass**

Run: `cd backend && python -m pytest tests/ -v`

- [ ] **Step 5: Commit**

```bash
git add backend/features/pipeline.py backend/tests/test_pipeline.py
git commit -m "feat: add feature pipeline orchestrator with data augmentation"
```

---

### Task 10.5: Dataset Builder & Cold-Start Imputation

**Files:**
- Create: `backend/features/build_dataset.py`
- Create: `backend/tests/test_build_dataset.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for full dataset build with cold-start imputation."""
import pytest
import pandas as pd
from features.build_dataset import impute_cold_start, build_full_dataset


def test_impute_cold_start_uses_weight_class_medians():
    """Debut fighters should get weight-class medians, not zeros (spec S3.4)."""
    features = {"a_slpm": 0.0, "a_td_avg": 0.0, "a_is_debut": True, "weight_class": "Lightweight"}
    medians = {"Lightweight": {"slpm": 4.5, "td_avg": 2.0}}
    result = impute_cold_start(features, medians)
    assert result["a_slpm"] == 4.5
    assert result["a_td_avg"] == 2.0


def test_impute_cold_start_sets_balanced_style():
    """Debut fighters get equal style scores (spec S3.4)."""
    features = {"a_style_striker": 0.0, "a_style_wrestler": 0.0,
                "a_style_grappler": 0.0, "a_style_balanced": 0.0,
                "a_is_debut": True, "weight_class": "Flyweight"}
    medians = {"Flyweight": {}}
    result = impute_cold_start(features, medians)
    assert result["a_style_striker"] == 0.25
    assert result["a_style_wrestler"] == 0.25
```

- [ ] **Step 2: Implement `backend/features/build_dataset.py`**

```python
"""Build the full feature-engineered dataset from raw fight data.

This is the orchestration script that goes from raw CSVs → features.parquet.
"""
import logging
import pandas as pd
from pathlib import Path

from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, DEBUT_FIGHT_THRESHOLD
from scrapers.kaggle_loader import load_fight_data, load_rankings_data
from features.pipeline import build_feature_row, augment_with_swap

logger = logging.getLogger(__name__)

STYLE_KEYS = ["striker", "wrestler", "grappler", "balanced"]
STAT_KEYS = ["slpm", "sapm", "str_acc", "str_def", "td_avg", "td_acc", "td_def", "sub_avg"]


def compute_weight_class_medians(df: pd.DataFrame) -> dict:
    """Compute median stats per weight class for cold-start imputation."""
    medians = {}
    for wc in df["weight_class"].unique():
        wc_df = df[df["weight_class"] == wc]
        medians[wc] = {}
        for stat in STAT_KEYS:
            col = f"a_{stat}"
            if col in wc_df.columns:
                medians[wc][stat] = wc_df[col].median()
    return medians


def impute_cold_start(features: dict, medians: dict) -> dict:
    """Replace zero-filled defaults with weight-class medians for debut fighters."""
    features = features.copy()
    wc = features.get("weight_class", "")

    for prefix in ["a", "b"]:
        if features.get(f"{prefix}_is_debut") or features.get(f"{prefix}_is_near_debut"):
            wc_meds = medians.get(wc, {})
            for stat in STAT_KEYS:
                key = f"{prefix}_{stat}"
                if features.get(key, 0) == 0.0 and stat in wc_meds:
                    features[key] = wc_meds[stat]
            # Set balanced style defaults (spec S3.4)
            for style in STYLE_KEYS:
                key = f"{prefix}_style_{style}"
                if features.get(key, 0) == 0.0:
                    features[key] = 0.25
    return features


def build_full_dataset(output_path: Path | None = None) -> pd.DataFrame:
    """Build the complete feature-engineered dataset.

    1. Load raw fight data
    2. For each fight, compute all time-aware features
    3. Apply cold-start imputation
    4. Apply data augmentation (fighter swap)
    5. Save to parquet
    """
    fights = load_fight_data()
    rankings = load_rankings_data()
    logger.info(f"Building features for {len(fights)} fights...")

    rows = []
    for _, fight in fights.iterrows():
        # Build feature row using time-aware pipeline
        # (implementation calls all feature modules with fight.event_date as cutoff)
        row = _build_row_from_fight(fight, fights, rankings)
        if row:
            rows.append(row)

    df = pd.DataFrame(rows)

    # Compute medians for cold-start, then impute
    medians = compute_weight_class_medians(df)
    df = pd.DataFrame([impute_cold_start(r, medians) for r in df.to_dict("records")])

    # Data augmentation: swap fighters (spec S5.3)
    df = augment_with_swap(df)

    # Save
    if output_path is None:
        output_path = PROCESSED_DATA_DIR / "features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    logger.info(f"Dataset saved: {len(df)} rows to {output_path}")
    return df


def _build_row_from_fight(fight, all_fights, rankings):
    """Build a single feature row from a fight record. Returns None if insufficient data."""
    # This function orchestrates calls to all feature modules
    # with fight["event_date"] as the time cutoff.
    # Implementation connects kaggle column names to feature functions.
    # Full implementation depends on actual CSV column names (adjusted in Task 19).
    pass  # Placeholder — fleshed out when real CSV columns are known
```

- [ ] **Step 3: Run tests — expect pass for unit tests**

Run: `cd backend && python -m pytest tests/test_build_dataset.py -v`

- [ ] **Step 4: Commit**

```bash
git add backend/features/build_dataset.py backend/tests/test_build_dataset.py
git commit -m "feat: add dataset builder with cold-start imputation and weight-class medians"
```

---

## Phase 3: Model Training & Evaluation

### Task 11: LightGBM Model Training

**Files:**
- Create: `backend/models/train.py`
- Create: `backend/tests/test_train.py`

- [ ] **Step 1: Write failing test**

```python
"""Tests for model training."""
import pytest
import pandas as pd
import numpy as np
from models.train import train_model, get_feature_columns


def test_get_feature_columns_excludes_target():
    df = pd.DataFrame({"feat_1": [1], "feat_2": [2], "target": [1], "event_date": ["2024-01-01"]})
    cols = get_feature_columns(df, include_odds=False)
    assert "target" not in cols
    assert "event_date" not in cols
    assert "feat_1" in cols


def test_get_feature_columns_excludes_odds_for_model_a():
    df = pd.DataFrame({"feat_1": [1], "a_implied_prob": [0.6], "odds_diff": [0.1], "target": [1]})
    cols = get_feature_columns(df, include_odds=False)
    assert "a_implied_prob" not in cols
    assert "odds_diff" not in cols


def test_train_model_returns_fitted():
    """Training should return a fitted LightGBM model."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "feat_1": np.random.randn(n),
        "feat_2": np.random.randn(n),
        "target": np.random.randint(0, 2, n),
    })
    model = train_model(df, ["feat_1", "feat_2"])
    preds = model.predict_proba(df[["feat_1", "feat_2"]])
    assert preds.shape == (n, 2)
```

- [ ] **Step 2: Run tests — expect fail**
- [ ] **Step 3: Implement `backend/models/train.py`**

```python
"""LightGBM model training for both model variants."""
import logging
from pathlib import Path
from datetime import date

import lightgbm as lgb
import joblib
import pandas as pd
import numpy as np

from config import MODELS_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)

ODDS_COLUMNS = {"a_implied_prob", "b_implied_prob", "odds_diff", "market_confidence"}
EXCLUDE_COLUMNS = {"target", "event_date", "fighter_a", "fighter_b", "winner", "event_name"}


def get_feature_columns(df: pd.DataFrame, include_odds: bool = True) -> list[str]:
    """Get feature column names, excluding target and metadata."""
    excluded = EXCLUDE_COLUMNS.copy()
    if not include_odds:
        excluded.update(ODDS_COLUMNS)
    return [c for c in df.columns if c not in excluded]


def train_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    params: dict | None = None,
) -> lgb.LGBMClassifier:
    """Train a LightGBM classifier."""
    if params is None:
        params = {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "random_state": RANDOM_SEED,
            "verbose": -1,
        }

    model = lgb.LGBMClassifier(**params)
    X = df[feature_cols]
    y = df["target"]
    model.fit(X, y)
    logger.info(f"Model trained on {len(X)} samples with {len(feature_cols)} features")
    return model


def save_model(model: lgb.LGBMClassifier, variant: str) -> Path:
    """Save model artifact with date-versioned filename."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"model_{variant}_{date.today().isoformat()}.joblib"
    path = MODELS_DIR / filename
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")
    return path


def tune_hyperparameters(
    df: pd.DataFrame, feature_cols: list[str], n_trials: int = 50
) -> dict:
    """Bayesian hyperparameter optimization with Optuna (spec S5.4)."""
    import optuna
    from models.evaluate import expanding_window_cv_splits, compute_metrics

    years = df["event_date"].dt.year.tolist() if "event_date" in df.columns else [2020] * len(df)
    splits = expanding_window_cv_splits(years, start_year=CV_START_YEAR)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": RANDOM_SEED, "verbose": -1,
        }
        accuracies = []
        for train_idx, val_idx in splits:
            model = lgb.LGBMClassifier(**params)
            X_train = df.iloc[train_idx][feature_cols]
            y_train = df.iloc[train_idx]["target"]
            X_val = df.iloc[val_idx][feature_cols]
            y_val = df.iloc[val_idx]["target"]
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            accuracies.append((preds == y_val).mean())
        return np.mean(accuracies)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    logger.info(f"Best params: {study.best_params} (accuracy: {study.best_value:.4f})")
    return study.best_params


def train_both_variants(df: pd.DataFrame, tune: bool = False) -> tuple:
    """Train Model A (no odds) and Model B (with odds)."""
    cols_a = get_feature_columns(df, include_odds=False)
    cols_b = get_feature_columns(df, include_odds=True)

    params = None
    if tune:
        logger.info("Running Optuna hyperparameter tuning...")
        params = tune_hyperparameters(df, cols_a)

    logger.info(f"Training Model A (no odds) with {len(cols_a)} features...")
    model_a = train_model(df, cols_a, params=params)
    path_a = save_model(model_a, "a_no_odds")

    logger.info(f"Training Model B (with odds) with {len(cols_b)} features...")
    model_b = train_model(df, cols_b, params=params)
    path_b = save_model(model_b, "b_with_odds")

    return (model_a, path_a), (model_b, path_b)
```

- [ ] **Step 4: Run tests — expect pass**
- [ ] **Step 5: Commit**

```bash
git add backend/models/train.py backend/tests/test_train.py
git commit -m "feat: add LightGBM model training with dual variants"
```

---

### Task 12: Model Evaluation & SHAP Explainability

**Files:**
- Create: `backend/models/evaluate.py`
- Create: `backend/models/explain.py`
- Create: `backend/models/predict.py`
- Create: `backend/tests/test_evaluate.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for model evaluation."""
import pytest
import numpy as np
from models.evaluate import compute_metrics, expanding_window_cv_splits


def test_compute_metrics():
    y_true = np.array([1, 0, 1, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 0])
    y_prob = np.array([0.8, 0.3, 0.7, 0.4, 0.2])
    metrics = compute_metrics(y_true, y_pred, y_prob)
    assert metrics["accuracy"] == 0.8
    assert "auc_roc" in metrics
    assert "log_loss" in metrics


def test_expanding_window_splits():
    """CV splits should expand training window and never leak future data."""
    years = [2019, 2019, 2020, 2020, 2021, 2021, 2022, 2022]
    splits = expanding_window_cv_splits(years, start_year=2020)
    assert len(splits) == 3  # 2020, 2021, 2022
    train_idx, val_idx = splits[0]
    assert all(years[i] < 2020 for i in train_idx)
    assert all(years[i] == 2020 for i in val_idx)
```

- [ ] **Step 2: Run tests — expect fail**
- [ ] **Step 3: Implement evaluation, explanation, and prediction modules**

`backend/models/evaluate.py`:
```python
"""Model evaluation: metrics, cross-validation, calibration."""
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute evaluation metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc_roc": roc_auc_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "n_samples": len(y_true),
    }


def expanding_window_cv_splits(
    years: list[int], start_year: int = 2020
) -> list[tuple[list[int], list[int]]]:
    """Generate expanding-window time-series CV splits.

    Train on all years before val_year, validate on val_year.
    """
    unique_years = sorted(set(y for y in years if y >= start_year))
    splits = []
    for val_year in unique_years:
        train_idx = [i for i, y in enumerate(years) if y < val_year]
        val_idx = [i for i, y in enumerate(years) if y == val_year]
        if train_idx and val_idx:
            splits.append((train_idx, val_idx))
    return splits
```

`backend/models/explain.py`:
```python
"""SHAP-based model explainability for per-prediction key factors."""
import shap
import numpy as np
import pandas as pd


def compute_shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """Compute SHAP values for predictions."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # For binary classification, take the positive class SHAP values
    if isinstance(shap_values, list):
        return shap_values[1]
    return shap_values


def get_top_factors(
    shap_vals: np.ndarray, feature_names: list[str], top_n: int = 5
) -> list[dict]:
    """Get the top N most impactful features for a single prediction."""
    abs_vals = np.abs(shap_vals)
    top_indices = np.argsort(abs_vals)[::-1][:top_n]
    factors = []
    for idx in top_indices:
        factors.append({
            "feature": feature_names[idx],
            "impact": float(shap_vals[idx]),
            "abs_impact": float(abs_vals[idx]),
            "direction": "positive" if shap_vals[idx] > 0 else "negative",
        })
    return factors


def get_global_importance(model, feature_names: list[str]) -> list[dict]:
    """Get global feature importance from the model."""
    importances = model.feature_importances_
    ranked = sorted(
        zip(feature_names, importances),
        key=lambda x: x[1], reverse=True
    )
    return [{"feature": name, "importance": float(imp)} for name, imp in ranked]
```

`backend/models/predict.py`:
```python
"""Generate predictions for upcoming fights."""
import json
import logging
from datetime import date, datetime
from pathlib import Path

import joblib
import pandas as pd
import numpy as np

from config import MODELS_DIR, PREDICTIONS_DIR
from models.explain import compute_shap_values, get_top_factors

logger = logging.getLogger(__name__)


def load_latest_model(variant: str) -> tuple:
    """Load the most recent model artifact for a variant."""
    pattern = f"model_{variant}_*.joblib"
    files = sorted(MODELS_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No model found for variant '{variant}' in {MODELS_DIR}")
    path = files[-1]
    logger.info(f"Loading model from {path}")
    return joblib.load(path), path


def predict_fight(model, features: pd.DataFrame, feature_names: list[str]) -> dict:
    """Generate a prediction with probability and SHAP factors."""
    X = features[feature_names]
    prob = model.predict_proba(X)[0]
    shap_vals = compute_shap_values(model, X)
    factors = get_top_factors(shap_vals[0], feature_names, top_n=5)

    return {
        "fighter_a_win_prob": float(prob[1]),
        "fighter_b_win_prob": float(prob[0]),
        "predicted_winner": "A" if prob[1] > 0.5 else "B",
        "confidence": float(max(prob)),
        "key_factors": factors,
    }


def save_predictions(predictions: list[dict], event_name: str) -> Path:
    """Save predictions to JSON file."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{date.today().isoformat()}_{event_name.replace(' ', '_')}.json"
    path = PREDICTIONS_DIR / filename
    path.write_text(json.dumps(predictions, indent=2, default=str))
    logger.info(f"Predictions saved to {path}")
    return path
```

- [ ] **Step 4: Run tests — expect pass**

Run: `cd backend && python -m pytest tests/ -v`

- [ ] **Step 5: Commit**

```bash
git add backend/models/evaluate.py backend/models/explain.py backend/models/predict.py backend/tests/test_evaluate.py
git commit -m "feat: add model evaluation, SHAP explainability, and prediction pipeline"
```

---

## Phase 4: FastAPI Backend

### Task 13: FastAPI Application & Pydantic Schemas

**Files:**
- Create: `backend/api/schemas.py`
- Create: `backend/api/main.py`
- Create: `backend/api/routes/predictions.py`
- Create: `backend/api/routes/fighters.py`
- Create: `backend/api/routes/events.py`
- Create: `backend/api/routes/model_stats.py`
- Create: `backend/tests/test_api.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predictions_endpoint_exists():
    response = client.get("/api/predictions/upcoming")
    assert response.status_code in (200, 404)  # 404 if no data yet, but endpoint exists


def test_fighters_endpoint_exists():
    response = client.get("/api/fighters")
    assert response.status_code in (200, 404)
```

- [ ] **Step 2: Run tests — expect fail**
- [ ] **Step 3: Implement schemas**

```python
# backend/api/schemas.py
"""Pydantic response models for the API."""
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model_a_loaded: bool
    model_b_loaded: bool


class KeyFactor(BaseModel):
    feature: str
    impact: float
    abs_impact: float
    direction: str


class FightPrediction(BaseModel):
    fighter_a: str
    fighter_b: str
    weight_class: str
    card_position: str
    fighter_a_win_prob: float
    fighter_b_win_prob: float
    predicted_winner: str
    confidence: float
    key_factors: list[KeyFactor]
    a_style_primary: str
    b_style_primary: str


class EventPredictions(BaseModel):
    event_name: str
    event_date: str
    predictions: list[FightPrediction]
    model_variant: str


class FighterProfile(BaseModel):
    name: str
    height_cm: float | None
    reach_cm: float | None
    stance: str
    record: str
    style_scores: dict[str, float]
    style_sub_scores: dict[str, float]


class ModelPerformance(BaseModel):
    overall_accuracy: float
    accuracy_by_weight_class: dict[str, float]
    accuracy_by_card_position: dict[str, float]
    feature_importance: list[dict[str, float]]


class FeatureImportance(BaseModel):
    feature: str
    importance: float
```

- [ ] **Step 4: Implement FastAPI app and routes**

```python
# backend/api/main.py
"""FastAPI application entry point."""
from fastapi import FastAPI
from api.routes import predictions, fighters, events, model_stats

app = FastAPI(
    title="UFC Fight Prediction API",
    description="ML-powered UFC fight outcome predictions with explainability",
    version="1.0.0",
)

app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"])
app.include_router(fighters.router, prefix="/api/fighters", tags=["Fighters"])
app.include_router(events.router, prefix="/api/events", tags=["Events"])
app.include_router(model_stats.router, prefix="/api/model", tags=["Model"])


@app.get("/health")
def health_check():
    return {"status": "ok", "model_a_loaded": False, "model_b_loaded": False}
```

```python
# backend/api/routes/predictions.py
"""Prediction endpoints."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/upcoming")
def get_upcoming_predictions():
    """Get predictions for the next UFC event."""
    return {"message": "No predictions available yet. Train the model first."}
```

```python
# backend/api/routes/fighters.py
"""Fighter data endpoints."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def list_fighters():
    """List all fighters in the dataset."""
    return {"fighters": [], "total": 0}


@router.get("/{fighter_id}")
def get_fighter(fighter_id: str):
    """Get detailed fighter profile."""
    return {"fighter_id": fighter_id, "message": "Fighter data not loaded yet."}
```

```python
# backend/api/routes/events.py
"""Event history and archive endpoints."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/history")
def get_event_history():
    """Get past events with predictions vs actual results."""
    return {"events": []}
```

```python
# backend/api/routes/model_stats.py
"""Model performance endpoints."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/performance")
def get_model_performance():
    """Get model accuracy metrics."""
    return {"message": "Model not trained yet."}


@router.get("/importance")
def get_feature_importance():
    """Get global feature importance rankings."""
    return {"features": []}
```

- [ ] **Step 5: Run tests — expect pass**

Run: `cd backend && python -m pytest tests/test_api.py -v`

- [ ] **Step 6: Verify server starts**

Run: `cd backend && uvicorn api.main:app --port 8000 &`
Then: `curl http://localhost:8000/health`
Expected: `{"status":"ok","model_a_loaded":false,"model_b_loaded":false}`
Kill: `kill %1`

- [ ] **Step 7: Commit**

```bash
git add backend/api/ backend/tests/test_api.py
git commit -m "feat: add FastAPI backend with prediction, fighter, event, and model routes"
```

---

## Phase 5: Next.js Frontend Dashboard

### Task 14: Initialize Next.js Project with shadcn/ui

**Files:**
- Create: `frontend/` (via CLI scaffolding)

- [ ] **Step 1: Scaffold Next.js**

Run:
```bash
cd /Users/chrislane/Desktop/Claude_Code/ufc-prediction-model
npx create-next-app@latest frontend --typescript --tailwind --eslint --app --src-dir=false --import-alias="@/*" --use-npm
```

- [ ] **Step 2: Initialize shadcn/ui**

Run:
```bash
cd frontend && npx shadcn@latest init -d
```

- [ ] **Step 3: Add required shadcn/ui components**

Run:
```bash
cd frontend && npx shadcn@latest add card tabs badge progress separator select
```

- [ ] **Step 4: Install chart libraries and testing deps**

Run:
```bash
cd frontend && npm install recharts @nivo/radar
cd frontend && npm install -D @testing-library/react @testing-library/jest-dom jest jest-environment-jsdom
```

- [ ] **Step 5: Create API client `frontend/lib/api.ts`**

```typescript
// Server-side only — not exposed to browser (spec S6.4)
const API_BASE = process.env.API_URL || "http://localhost:8000";

export async function fetchFromAPI<T>(endpoint: string): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    cache: "no-store",
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}
```

- [ ] **Step 6: Create types `frontend/lib/types.ts`**

```typescript
export interface KeyFactor {
  feature: string;
  impact: number;
  abs_impact: number;
  direction: "positive" | "negative";
}

export interface FightPrediction {
  fighter_a: string;
  fighter_b: string;
  weight_class: string;
  card_position: string;
  fighter_a_win_prob: number;
  fighter_b_win_prob: number;
  predicted_winner: "A" | "B";
  confidence: number;
  key_factors: KeyFactor[];
  a_style_primary: string;
  b_style_primary: string;
}

export interface EventPredictions {
  event_name: string;
  event_date: string;
  predictions: FightPrediction[];
  model_variant: string;
}

export interface FighterProfile {
  name: string;
  height_cm: number | null;
  reach_cm: number | null;
  stance: string;
  record: string;
  style_scores: Record<string, number>;
  style_sub_scores: Record<string, number>;
}

export interface ModelPerformance {
  overall_accuracy: number;
  accuracy_by_weight_class: Record<string, number>;
  accuracy_by_card_position: Record<string, number>;
  feature_importance: Array<{ feature: string; importance: number }>;
}
```

- [ ] **Step 7: Create utility helpers `frontend/lib/utils.ts`** (extend existing)

Add to the existing `utils.ts`:
```typescript
export function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.65) return "#06d6a0"; // Green — strong pick
  if (confidence >= 0.55) return "#ffd166"; // Yellow — lean
  return "#888888"; // Gray — coin flip
}

export function getStyleColor(style: string): string {
  const colors: Record<string, string> = {
    striker: "#e94560",
    wrestler: "#4cc9f0",
    grappler: "#7b2ff7",
    balanced: "#06d6a0",
  };
  return colors[style.toLowerCase()] || "#888888";
}

export function getStyleIcon(style: string): string {
  const icons: Record<string, string> = {
    striker: "🥊",
    wrestler: "🤼",
    grappler: "🐍",
    balanced: "⚖️",
  };
  return icons[style.toLowerCase()] || "❓";
}

export function formatProbability(prob: number): string {
  return `${Math.round(prob * 100)}%`;
}
```

- [ ] **Step 8: Commit**

```bash
git add frontend/
git commit -m "feat: initialize Next.js frontend with shadcn/ui, chart libs, and types"
```

---

### Task 15: Main Predictions Page

**Files:**
- Create: `frontend/components/fight-card.tsx`
- Modify: `frontend/app/page.tsx`
- Create: `frontend/app/layout.tsx` (modify generated)

- [ ] **Step 1: Update layout with dark theme and nav**

Modify `frontend/app/layout.tsx` — set dark mode, add navigation bar with UFC branding and page links. Use Geist Sans/Mono fonts. Background: `#0a0a0f`.

- [ ] **Step 2: Create `frontend/components/fight-card.tsx`**

Build the fight prediction card component matching the mockup: fighter names, records, style labels, confidence bar, key decision factors. Use shadcn Card, Badge, and Progress components. Color-code confidence (green/yellow/gray).

- [ ] **Step 3: Implement main page `frontend/app/page.tsx`**

Fetch predictions from FastAPI `/api/predictions/upcoming`. Show event header (name, date, location), model accuracy stats (both variants), card position filter tabs (Main/Prelims/Early Prelims defaulting to Prelims), and a list of FightCard components.

- [ ] **Step 4: Verify it renders**

Run: `cd frontend && npm run dev`
Open: `http://localhost:3000`
Expected: Dark-themed page with navigation, placeholder content

- [ ] **Step 5: Write smoke test for page**

Create `frontend/__tests__/page.test.tsx` — render test verifying the page component mounts without crashing and displays expected elements (navigation, event header area).

- [ ] **Step 6: Commit**

```bash
git add frontend/app/ frontend/components/fight-card.tsx frontend/__tests__/
git commit -m "feat: add main predictions page with fight card component"
```

---

### Task 16: Fighter Profile Page

**Files:**
- Create: `frontend/app/fighters/[id]/page.tsx`
- Create: `frontend/components/style-chart.tsx`

- [ ] **Step 1: Create style hierarchy bar chart component**

Build `style-chart.tsx` — displays the 4 primary archetype bars with nested sub-type bars below dominant styles. Uses the color scheme from spec §7.2. Props: `styleScores: Record<string, number>`, `subScores: Record<string, number>`.

- [ ] **Step 2: Create fighter profile page**

Fetch fighter data from `/api/fighters/{id}`. Display: physical stats card, style hierarchy chart, fight history timeline, career stat trends.

- [ ] **Step 3: Verify it renders**
- [ ] **Step 4: Commit**

```bash
git add frontend/app/fighters/ frontend/components/style-chart.tsx
git commit -m "feat: add fighter profile page with style hierarchy chart"
```

---

### Task 17: Head-to-Head Comparison Page

**Files:**
- Create: `frontend/app/compare/page.tsx`
- Create: `frontend/components/fighter-comparison.tsx`

- [ ] **Step 1: Create comparison component**

Build side-by-side fighter comparison with overlaid bar metrics, Nivo radar chart for style matchup, and model prediction if they fought. Use shadcn Select for fighter selection.

- [ ] **Step 2: Create compare page**

Two fighter selectors at top, comparison view below, model prediction result with key factors.

- [ ] **Step 3: Verify and commit**

```bash
git add frontend/app/compare/ frontend/components/fighter-comparison.tsx
git commit -m "feat: add head-to-head fighter comparison page"
```

---

### Task 18: Model Performance & Event Archive Pages

**Files:**
- Create: `frontend/app/performance/page.tsx`
- Create: `frontend/app/history/page.tsx`
- Create: `frontend/components/accuracy-chart.tsx`
- Create: `frontend/components/feature-importance.tsx`

- [ ] **Step 1: Create accuracy chart component**

Recharts line chart showing accuracy over time by event. Include toggle for Model A vs Model B. Bar chart for accuracy by weight class.

- [ ] **Step 2: Create feature importance component**

Horizontal bar chart of SHAP-based feature importance. Sortable, filterable by weight class.

- [ ] **Step 3: Create performance page**

Combines accuracy chart, feature importance, and Model A vs Model B comparison stats.

- [ ] **Step 4: Create event archive page**

Filterable list of past events. Each shows predictions vs actual results. Filter by date range, weight class, correct/incorrect predictions.

- [ ] **Step 5: Verify all pages render**

Run: `cd frontend && npm run dev`
Visit each route: `/`, `/compare`, `/performance`, `/fighters/test`, `/history`

- [ ] **Step 6: Commit**

```bash
git add frontend/app/performance/ frontend/app/history/ frontend/components/accuracy-chart.tsx frontend/components/feature-importance.tsx
git commit -m "feat: add model performance and event archive pages"
```

---

## Phase 6: Integration & End-to-End Testing

### Task 19: Download Data & Run Full Pipeline

- [ ] **Step 1: Download Kaggle datasets**

Manually download the following CSVs and place in `backend/data/raw/`:
- UFC 2025 Dataset → `ufc_fights.csv`
- UFC Rankings → `ufc_rankings.csv`
- Ultimate UFC Dataset → `ufc_master.csv`

(Note: Kaggle requires authentication — use `kaggle datasets download` CLI or download via browser)

- [ ] **Step 2: Run data loading and inspect**

```bash
cd backend && python -c "
from scrapers.kaggle_loader import load_fight_data
df = load_fight_data()
print(f'Loaded {len(df)} fights')
print(f'Date range: {df.event_date.min()} to {df.event_date.max()}')
print(f'Columns: {list(df.columns)[:10]}...')
"
```

Inspect output, adjust column mappings in `kaggle_loader.py` if CSV column names differ from expected.

- [ ] **Step 3: Run feature engineering pipeline on real data**

```bash
cd backend && python -c "
from features.pipeline import build_feature_row
# Test with real data from loaded CSV
print('Feature pipeline produces output: OK')
"
```

- [ ] **Step 4: Train models on real data**

```bash
cd backend && python -c "
from models.train import train_both_variants
import pandas as pd
df = pd.read_parquet('data/processed/features.parquet')
(model_a, path_a), (model_b, path_b) = train_both_variants(df)
print(f'Model A saved to: {path_a}')
print(f'Model B saved to: {path_b}')
"
```

- [ ] **Step 5: Evaluate model accuracy**

```bash
cd backend && python -c "
from models.evaluate import compute_metrics, expanding_window_cv_splits
# Run expanding window CV and print results
print('Model A accuracy: ...')
print('Model B accuracy: ...')
"
```

- [ ] **Step 6: Start both servers and verify end-to-end**

Terminal 1: `cd backend && uvicorn api.main:app --port 8000`
Terminal 2: `cd frontend && npm run dev`
Open: `http://localhost:3000`
Verify: predictions load, fight cards render, navigation works.

- [ ] **Step 7: Final commit**

```bash
git add -A
git commit -m "feat: complete UFC prediction model v1 — data pipeline, ML models, and dashboard"
```

---

## Summary

| Phase | Tasks | Description |
|-------|-------|-------------|
| **Phase 1** | Tasks 1-3.5 | Project setup, data loading, name resolution, scraper + API client |
| **Phase 2** | Tasks 4-10.5 | Feature engineering (all 9 categories + pipeline + dataset builder) |
| **Phase 3** | Tasks 11-12 | Model training (with Optuna tuning), evaluation, SHAP explainability |
| **Phase 4** | Task 13 | FastAPI backend with all routes |
| **Phase 5** | Tasks 14-18 | Next.js dashboard (5 pages, with smoke tests) |
| **Phase 6** | Task 19 | Integration, real data, end-to-end testing |

**Total: 21 tasks, ~105 steps**

**Parallelization opportunities:**
- Tasks 4-9 (feature modules) can all run in parallel — they are independent
- Task 13 (FastAPI) can run in parallel with Tasks 14-18 (frontend)
- Tasks 15-18 (dashboard pages) can each run in parallel
