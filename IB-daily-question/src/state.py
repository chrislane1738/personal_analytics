import fcntl
import json
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

DEFAULT = {"telegram_offset": 0, "pending": None, "history": []}


def load(path: Path) -> dict:
    if not path.exists():
        return dict(DEFAULT, history=[])
    try:
        with path.open("r") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                return json.load(f)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except (json.JSONDecodeError, ValueError):
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        backup = path.with_suffix(f".json.bak.{ts}")
        shutil.copy(path, backup)
        return dict(DEFAULT, history=[])


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            json.dump(data, f, indent=2)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    tmp.replace(path)


@contextmanager
def mutate(path: Path) -> Iterator[dict]:
    """Read-modify-write with exclusive lock held the entire time."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    with path.open("r+") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            raw = f.read()
            data = json.loads(raw) if raw.strip() else dict(DEFAULT, history=[])
            yield data
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=2)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
