import json
from unittest.mock import patch, Mock
import pytest

from src import grade


def _anthropic_response(content_text: str):
    msg = Mock()
    msg.content = [Mock(text=content_text)]
    return msg


VALID_GRADE = {
    "score": 78, "letter": "B+",
    "nailed": ["Got the formula"],
    "missed": ["Forgot tax shield"],
    "feedback": "Solid structure.",
}


@patch("src.grade.Anthropic")
def test_grade_returns_parsed_json(mock_anthropic_cls):
    mock_client = Mock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.return_value = _anthropic_response(json.dumps(VALID_GRADE))

    out = grade.grade(
        question="How do you calc WACC?",
        rubric={"must_hit": ["formula"], "scoring_weights": {}},
        transcript="WACC equals...",
        api_key="sk-ant-1",
    )

    assert out == VALID_GRADE
    call = mock_client.messages.create.call_args
    assert call.kwargs["model"] == "claude-haiku-4-5-20251001"


@patch("src.grade.Anthropic")
def test_grade_retries_on_malformed_then_succeeds(mock_anthropic_cls):
    mock_client = Mock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.side_effect = [
        _anthropic_response("not json at all"),
        _anthropic_response(json.dumps(VALID_GRADE)),
    ]

    out = grade.grade("Q?", {}, "answer", api_key="sk-ant-1")
    assert out == VALID_GRADE
    assert mock_client.messages.create.call_count == 2


@patch("src.grade.Anthropic")
def test_grade_raises_after_two_failures(mock_anthropic_cls):
    mock_client = Mock()
    mock_anthropic_cls.return_value = mock_client
    mock_client.messages.create.return_value = _anthropic_response("still not json")
    with pytest.raises(grade.GraderError):
        grade.grade("Q?", {}, "answer", api_key="sk-ant-1")


@patch("src.grade.Anthropic")
def test_grade_extracts_json_from_code_fence(mock_anthropic_cls):
    mock_client = Mock()
    mock_anthropic_cls.return_value = mock_client
    fenced = f"```json\n{json.dumps(VALID_GRADE)}\n```"
    mock_client.messages.create.return_value = _anthropic_response(fenced)
    out = grade.grade("Q?", {}, "answer", api_key="sk-ant-1")
    assert out == VALID_GRADE
