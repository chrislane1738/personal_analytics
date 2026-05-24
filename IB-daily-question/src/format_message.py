def format_question(q: dict) -> str:
    return (
        f"*Today's IB question* — _{q['category']}_\n\n"
        f"{q['question']}\n\n"
        f"_Reply with a voice message — answer like you're talking to an interviewer._"
    )


def _bullets(items: list[str], empty_text: str) -> str:
    if not items:
        return f"  _{empty_text}_"
    return "\n".join(f"  • {x}" for x in items)


def format_regrade(prev: dict) -> str:
    prior = prev.get("prior_grade") or {}
    score, letter = prior.get("score"), prior.get("letter")
    header = "*Regrading — let's try again*"
    if score is not None and letter:
        header += f" _(previous: {letter} {score}/100)_"
    return (
        f"{header} — _{prev['category']}_\n\n"
        f"{prev['question']}\n\n"
        f"_Reply with a voice message._"
    )


def format_represent(q: dict) -> str:
    return (
        f"*Re-presenting your current question* — _{q['category']}_\n\n"
        f"{q['question']}\n\n"
        f"_Reply with a voice message._"
    )


def format_result(grade: dict) -> str:
    return (
        f"*Grade: {grade['letter']} ({grade['score']}/100)*\n\n"
        f"✓ *What you nailed*\n"
        f"{_bullets(grade['nailed'], 'Nothing stood out.')}\n\n"
        f"✗ *What you missed*\n"
        f"{_bullets(grade['missed'], 'Nothing significant missed.')}\n\n"
        f"*Feedback*\n{grade['feedback']}"
    )
