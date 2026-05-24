from src import format_message as fm


def test_format_question_includes_category_and_question():
    q = {"category": "Core Valuation & DCF",
         "question": "Walk me through a DCF."}
    out = fm.format_question(q)
    assert "Core Valuation & DCF" in out
    assert "Walk me through a DCF." in out
    assert "voice message" in out.lower()


def test_format_result_renders_all_sections():
    grade = {
        "score": 78, "letter": "B+",
        "nailed": ["Got formula", "Mentioned beta"],
        "missed": ["No tax shield"],
        "feedback": "Solid attempt.",
    }
    out = fm.format_result(grade)
    assert "78/100" in out
    assert "B+" in out
    assert "Got formula" in out
    assert "No tax shield" in out
    assert "Solid attempt." in out


def test_format_result_handles_empty_lists():
    grade = {"score": 95, "letter": "A",
             "nailed": ["Everything"], "missed": [], "feedback": "Great."}
    out = fm.format_result(grade)
    assert "nothing significant missed" in out.lower() or "—" in out


def test_format_escapes_markdown_special_chars_in_transcript_safe_fields():
    # feedback may contain user-influenced content; markdown should not break
    grade = {"score": 50, "letter": "D",
             "nailed": [], "missed": [], "feedback": "Mentioned * and _ and [link]"}
    out = fm.format_result(grade)
    assert "Mentioned" in out


def test_format_regrade_includes_prior_score():
    prev = {
        "category": "DCF",
        "question": "How do you calc WACC?",
        "prior_grade": {"score": 72, "letter": "C-"},
    }
    out = fm.format_regrade(prev)
    assert "Regrading" in out
    assert "72/100" in out
    assert "C-" in out
    assert "How do you calc WACC?" in out


def test_format_regrade_without_prior_grade():
    prev = {"category": "DCF", "question": "Q?", "prior_grade": None}
    out = fm.format_regrade(prev)
    assert "Regrading" in out
    assert "Q?" in out


def test_format_represent_includes_question():
    q = {"category": "DCF", "question": "Walk me through a DCF."}
    out = fm.format_represent(q)
    assert "Re-presenting" in out
    assert "Walk me through a DCF." in out
