import time
import requests

BASE = "https://api.telegram.org"


def _post(url: str, json: dict, max_retries: int = 3, backoff: float = 2.0):
    last_exc = None
    for attempt in range(max_retries):
        r = requests.post(url, json=json, timeout=30)
        try:
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            if attempt < max_retries - 1:
                time.sleep(backoff * (2 ** attempt))
    raise last_exc


def send_message(token: str, chat_id: str, text: str,
                 max_retries: int = 3, backoff: float = 2.0) -> int:
    url = f"{BASE}/bot{token}/sendMessage"
    r = _post(url, {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
              max_retries=max_retries, backoff=backoff)
    return r.json()["result"]["message_id"]


def get_updates(token: str, offset: int, timeout: int = 30) -> list[dict]:
    url = f"{BASE}/bot{token}/getUpdates"
    r = requests.get(
        url,
        params={"offset": offset, "timeout": timeout, "allowed_updates": ["message"]},
        timeout=timeout + 5,
    )
    r.raise_for_status()
    return r.json().get("result", [])


def download_voice(token: str, file_id: str) -> bytes:
    meta = requests.get(f"{BASE}/bot{token}/getFile",
                        params={"file_id": file_id}, timeout=30)
    meta.raise_for_status()
    file_path = meta.json()["result"]["file_path"]
    dl = requests.get(f"{BASE}/file/bot{token}/{file_path}", timeout=60)
    dl.raise_for_status()
    return dl.content
