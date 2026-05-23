import json
import threading
from pathlib import Path

from src import state


def test_load_returns_defaults_when_missing(tmp_path):
    p = tmp_path / "state.json"
    s = state.load(p)
    assert s == {"telegram_offset": 0, "pending": None, "history": []}


def test_save_then_load_roundtrip(tmp_path):
    p = tmp_path / "state.json"
    s = state.load(p)
    s["telegram_offset"] = 42
    state.save(p, s)
    assert state.load(p) == s


def test_mutate_applies_under_lock(tmp_path):
    p = tmp_path / "state.json"
    state.save(p, {"telegram_offset": 0, "pending": None, "history": []})

    def inc():
        for _ in range(100):
            with state.mutate(p) as s:
                s["telegram_offset"] += 1

    threads = [threading.Thread(target=inc) for _ in range(5)]
    for t in threads: t.start()
    for t in threads: t.join()

    final = state.load(p)
    assert final["telegram_offset"] == 500


def test_load_corrupt_backs_up_and_resets(tmp_path):
    p = tmp_path / "state.json"
    p.write_text("{not json")
    s = state.load(p)
    assert s == {"telegram_offset": 0, "pending": None, "history": []}
    # backup file exists
    backups = list(tmp_path.glob("state.json.bak.*"))
    assert len(backups) == 1
