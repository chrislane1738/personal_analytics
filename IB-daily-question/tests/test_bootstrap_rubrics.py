import json
from unittest.mock import patch, Mock
from pathlib import Path

from scripts import bootstrap_rubrics as br


def test_parse_docx_extracts_categories_and_questions(tmp_path, monkeypatch):
    """The parser should identify section headers vs question lines."""
    # We mock the docx unzip layer by patching the extractor
    monkeypatch.setattr(br, "_extract_text", lambda p: [
        "Core Valuation & DCF",
        "Understanding how to value a company is the core of the job.",
        "Walk me through a Discounted Cash Flow (DCF) model.",
        "How do you calculate WACC?",
        "Mergers & Acquisitions (M&A)",
        "Walk me through a basic merger model.",
    ])
    items = br.parse_docx(Path("ignored"))
    cats = [i["category"] for i in items]
    qs = [i["question"] for i in items]
    assert "Core Valuation & DCF" in cats
    assert "Mergers & Acquisitions (M&A)" in cats
    assert "Walk me through a Discounted Cash Flow (DCF) model." in qs
    assert "How do you calculate WACC?" in qs
    assert "Walk me through a basic merger model." in qs
    # description line should NOT be a question
    assert all("core of the job" not in q for q in qs)


def test_slug_id_is_stable():
    assert br.slug_id("How do you calculate WACC?") == br.slug_id("How do you calculate WACC?")
    assert br.slug_id("DCF model") != br.slug_id("LBO model")


@patch("scripts.bootstrap_rubrics.Anthropic")
def test_generate_rubric_calls_sonnet_and_parses(mock_anth_cls):
    mock_client = Mock()
    mock_anth_cls.return_value = mock_client
    rubric = {
        "must_hit": ["formula"],
        "good_to_hit": ["beta"],
        "common_pitfalls": ["book values"],
        "scoring_weights": {"must_hit": 0.65, "good_to_hit": 0.20, "clarity_structure": 0.15},
    }
    mock_client.messages.create.return_value = Mock(
        content=[Mock(text=json.dumps(rubric))]
    )
    out = br.generate_rubric("How do you calculate WACC?", "DCF", api_key="sk-ant-1")
    assert out == rubric


def test_skip_existing_when_not_regenerating(tmp_path):
    existing_id = br.slug_id("Q?")
    existing = [{"id": existing_id, "category": "DCF", "difficulty": "fundamental",
                 "question": "Q?", "rubric": {"must_hit": []}}]
    out_path = tmp_path / "questions.json"
    out_path.write_text(json.dumps(existing))

    items = [{"category": "DCF", "question": "Q?"}]  # same Q
    fake_gen = Mock()
    br.build(items, out_path, api_key="sk", regenerate=False, generator=fake_gen)
    fake_gen.assert_not_called()
    final = json.loads(out_path.read_text())
    assert final == existing
