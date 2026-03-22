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
    result = matcher.match("Charles oliveira")
    assert result == "Charles Oliveira"

def test_no_match_returns_original():
    matcher = FighterNameMatcher(aliases={}, known_fighters=[])
    result = matcher.match("Completely Unknown Fighter")
    assert result == "Completely Unknown Fighter"

def test_low_confidence_logged(matcher, caplog):
    """Matches below 95% similarity should be logged for review."""
    import logging
    with caplog.at_level(logging.WARNING):
        result = matcher.match("Charles Olivera", known_fighters=["Charles Oliveira"])
    assert result == "Charles Oliveira"
    assert any("Low-confidence" in msg for msg in caplog.messages)
