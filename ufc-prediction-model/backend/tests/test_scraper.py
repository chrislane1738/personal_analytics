"""Tests for UFC API client with caching and rate limiting."""
import pytest
import json
from pathlib import Path
from scrapers.api_client import UFCApiClient

def test_cached_response_used_when_available(tmp_path):
    cache_file = tmp_path / "fighters_test.json"
    cache_file.write_text(json.dumps({"name": "Cached Fighter"}))
    client = UFCApiClient(cache_dir=tmp_path)
    result = client._get_cached("fighters_test")
    assert result == {"name": "Cached Fighter"}

def test_scraper_fail_loud():
    from scrapers.ufcstats_scraper import scrape_fight_data
    with pytest.raises(Exception):
        scrape_fight_data(invalid_arg=True)
