"""One-time: parse the IB interview questions .docx and generate rubrics for each
question using Claude Sonnet. Writes data/questions.json.

Usage:
    python -m scripts.bootstrap_rubrics
    python -m scripts.bootstrap_rubrics --regenerate    # overwrite existing
"""
import argparse
import hashlib
import json
import re
import sys
import time
import zipfile
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
import os

REPO_ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = REPO_ROOT / "Investment Banking Interview Questions.docx"
OUT_PATH = REPO_ROOT / "data" / "questions.json"
MODEL = "claude-sonnet-4-6"


# --- .docx parsing ---

_KNOWN_CATEGORIES = {
    "The Absolute Fundamentals (Accounting & Enterprise Value)",
    "Core Valuation & DCF",
    "Advanced Accounting & Working Capital",
    "Mergers & Acquisitions (M&A)",
    "Leveraged Buyouts (LBO)",
}


def _extract_text(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8")
    # Unescape and strip tags
    xml = xml.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = re.sub(r"<[^>]+>", "\n", xml)
    return [line.strip() for line in text.splitlines() if line.strip()]


def parse_docx(path: Path) -> list[dict]:
    """Return list of {category, question}. Headers identified by exact match
    against _KNOWN_CATEGORIES; description lines (non-question text under a header)
    are filtered out by requiring the line to look like a question (ends with ?
    OR begins with a question-starter)."""
    lines = _extract_text(path)
    out = []
    current = None
    for line in lines:
        if line in _KNOWN_CATEGORIES:
            current = line
            continue
        if current is None:
            continue
        if _looks_like_question(line):
            out.append({"category": current, "question": line})
    return out


_QUESTION_STARTS = ("Walk me through", "A company", "If ", "When ", "Why ",
                    "How ", "What ", "Rank ", "Is ", "Can ")


def _looks_like_question(s: str) -> bool:
    if s.endswith("?"):
        return True
    return any(s.startswith(p) for p in _QUESTION_STARTS)


# --- ID generation ---

def slug_id(question: str) -> str:
    h = hashlib.sha1(question.encode("utf-8")).hexdigest()[:8]
    return f"q-{h}"


# --- Rubric generation ---

SYSTEM_RUBRIC = """You are a senior investment banking interviewer building a grading rubric for an entry-level analyst's verbal answer to a technical IB interview question.

Return ONLY a JSON object with this exact shape — no preamble, no markdown fences:

{
  "must_hit": ["<concrete point a strong answer MUST cover>", ...],
  "good_to_hit": ["<bonus point that distinguishes a top answer>", ...],
  "common_pitfalls": ["<frequent candidate mistake on this question>", ...],
  "scoring_weights": {"must_hit": 0.65, "good_to_hit": 0.20, "clarity_structure": 0.15}
}

must_hit should have 3-6 items, good_to_hit 2-4, common_pitfalls 2-4.

Each rubric point should be specific and concrete (e.g., "WACC formula with (E/V)·Re + (D/V)·Rd·(1−t)" not "WACC formula"). Phrase as scoring criteria, not as the answer itself."""


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


def generate_rubric(question: str, category: str, api_key: str) -> dict:
    client = Anthropic(api_key=api_key)
    user = f"Category: {category}\n\nQuestion: {question}\n\nReturn the rubric JSON now."
    resp = client.messages.create(
        model=MODEL, max_tokens=1024,
        system=SYSTEM_RUBRIC,
        messages=[{"role": "user", "content": user}],
    )
    return _extract_json(resp.content[0].text)


# --- Build orchestration ---

def build(items: list[dict], out_path: Path, api_key: str,
          regenerate: bool, generator=None) -> None:
    """Build/refresh questions.json. `generator` is injectable for tests."""
    gen = generator or (lambda q, c: generate_rubric(q, c, api_key))

    existing = []
    if out_path.exists():
        existing = json.loads(out_path.read_text())
    by_id = {e["id"]: e for e in existing}

    for item in items:
        qid = slug_id(item["question"])
        if qid in by_id and not regenerate:
            print(f"[skip] {qid}: {item['question'][:60]}")
            continue
        print(f"[gen]  {qid}: {item['question'][:60]}")
        rubric = gen(item["question"], item["category"])
        by_id[qid] = {
            "id": qid,
            "category": item["category"],
            "difficulty": "fundamental",
            "question": item["question"],
            "rubric": rubric,
        }
        time.sleep(0.5)  # gentle rate limit

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(list(by_id.values()), indent=2))
    print(f"Wrote {len(by_id)} questions to {out_path}")


def main() -> int:
    load_dotenv(REPO_ROOT / ".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Missing ANTHROPIC_API_KEY", file=sys.stderr)
        return 1

    ap = argparse.ArgumentParser()
    ap.add_argument("--regenerate", action="store_true",
                    help="Overwrite rubrics for questions that already exist")
    ap.add_argument("--docx", type=Path, default=DOC_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    items = parse_docx(args.docx)
    print(f"Parsed {len(items)} questions from {args.docx.name}")
    build(items, args.out, api_key=api_key, regenerate=args.regenerate)
    return 0


if __name__ == "__main__":
    sys.exit(main())
