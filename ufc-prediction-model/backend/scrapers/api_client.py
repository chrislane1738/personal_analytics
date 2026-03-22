"""REST client for ufcapi.aristotle.me with caching and rate limiting."""
import json
import logging
import time
from pathlib import Path
import requests
from config import UFC_API_BASE, UFC_API_DAILY_LIMIT, CACHE_DIR

logger = logging.getLogger(__name__)

class UFCApiClient:
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
        if cache_key:
            cached = self._get_cached(cache_key)
            if cached:
                return cached
        if self._request_count >= UFC_API_DAILY_LIMIT:
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
