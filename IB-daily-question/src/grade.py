import json
import re
from anthropic import Anthropic

MODEL = "claude-haiku-4-5-20251001"


class GraderError(Exception):
    pass


SYSTEM_PROMPT = """You are a VP-level investment banking interviewer grading a candidate's verbal answer to a technical interview question. You grade like a real MD/VP would: technical correctness matters most, but clarity and structure count too.

You will return ONLY a JSON object — no preamble, no markdown fences, no commentary — with this exact shape:

{
  "score": <integer 0-100>,
  "letter": "<A+|A|A-|B+|B|B-|C+|C|C-|D+|D|D-|F>",
  "nailed": ["<concrete point the candidate correctly hit>", ...],
  "missed": ["<concrete point the candidate missed or was unclear on>", ...],
  "feedback": "<2-4 sentences of overall feedback in an interviewer's voice>"
}

Scoring weights for this question:
- must_hit points: 65% of score (each one missed is significant)
- good_to_hit points: 20% of score (bonus, not required)
- clarity & structure: 15% of score

Letter mapping: 93+=A, 90-92=A-, 87-89=B+, 83-86=B, 80-82=B-, 77-79=C+, 73-76=C, 70-72=C-, 67-69=D+, 63-66=D, 60-62=D-, <60=F. (A+ reserved for genuinely exceptional answers >=97.)"""


USER_TEMPLATE = """QUESTION:
{question}

RUBRIC:
must_hit:
{must_hit}

good_to_hit:
{good_to_hit}

common_pitfalls:
{pitfalls}

CANDIDATE'S TRANSCRIBED ANSWER:
{transcript}

Return the JSON object now."""


def _bullets(items: list[str] | None) -> str:
    if not items:
        return "(none)"
    return "\n".join(f"- {x}" for x in items)


def _extract_json(text: str) -> dict:
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try fenced code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # Try first {...} blob
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise json.JSONDecodeError("no JSON found", text, 0)


def grade(question: str, rubric: dict, transcript: str, api_key: str) -> dict:
    client = Anthropic(api_key=api_key)
    user = USER_TEMPLATE.format(
        question=question,
        must_hit=_bullets(rubric.get("must_hit")),
        good_to_hit=_bullets(rubric.get("good_to_hit")),
        pitfalls=_bullets(rubric.get("common_pitfalls")),
        transcript=transcript,
    )

    for attempt in range(2):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        text = resp.content[0].text
        try:
            return _extract_json(text)
        except (json.JSONDecodeError, ValueError):
            if attempt == 0:
                user += "\n\nREMINDER: Return ONLY the JSON object. No prose, no fences."
                continue
            raise GraderError(f"Could not parse grader output: {text[:200]}")
