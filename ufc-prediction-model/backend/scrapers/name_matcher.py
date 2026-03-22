"""Fighter name matching with fuzzy matching and alias support."""
import json
import logging
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein
from config import ALIASES_PATH, FUZZY_MATCH_THRESHOLD, FUZZY_REVIEW_THRESHOLD

logger = logging.getLogger(__name__)


def _levenshtein_scorer(s1: str, s2: str, **kwargs) -> float:
    """Levenshtein normalized similarity scaled to 0-100."""
    return Levenshtein.normalized_similarity(s1, s2) * 100


class FighterNameMatcher:
    def __init__(self, aliases: dict[str, str] | None = None, known_fighters: list[str] | None = None):
        if aliases is None:
            aliases = self._load_aliases()
        self.aliases = aliases
        # Seed known_fighters from alias canonical values plus any explicitly provided names
        alias_values = list(dict.fromkeys(aliases.values()))  # deduplicated alias targets
        provided = known_fighters or []
        combined = list(dict.fromkeys(alias_values + provided))  # alias values first, then provided
        self.known_fighters = combined

    @staticmethod
    def _load_aliases() -> dict[str, str]:
        if ALIASES_PATH.exists():
            data = json.loads(ALIASES_PATH.read_text())
            return data.get("aliases", {})
        return {}

    def match(self, name: str, known_fighters: list[str] | None = None) -> str:
        fighters = known_fighters if known_fighters is not None else self.known_fighters
        # 1. Alias lookup
        if name in self.aliases:
            return self.aliases[name]
        # 2. Exact match (case-insensitive)
        name_lower = name.lower()
        for f in fighters:
            if f.lower() == name_lower:
                return f
        # 3. Fuzzy match
        if fighters:
            result = process.extractOne(
                name, fighters, scorer=_levenshtein_scorer, score_cutoff=FUZZY_MATCH_THRESHOLD
            )
            if result:
                matched_name, score, _ = result
                if score < FUZZY_REVIEW_THRESHOLD:
                    logger.warning(
                        f"Low-confidence match ({score:.0f}%): '{name}' → '{matched_name}'."
                        " Review and add to aliases if correct."
                    )
                return matched_name
        return name
