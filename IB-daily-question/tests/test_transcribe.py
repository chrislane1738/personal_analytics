from unittest.mock import patch, Mock
from src import transcribe


@patch("src.transcribe.OpenAI")
def test_whisper_returns_transcript_text(mock_openai_cls):
    mock_client = Mock()
    mock_openai_cls.return_value = mock_client
    mock_client.audio.transcriptions.create.return_value = Mock(text="hello world")

    out = transcribe.whisper(b"oggbytes", api_key="sk-1")

    assert out == "hello world"
    call = mock_client.audio.transcriptions.create.call_args
    assert call.kwargs["model"] == "whisper-1"
    # file is passed as (filename, bytes, mime) tuple
    assert call.kwargs["file"][0].endswith(".ogg")
    assert call.kwargs["file"][1] == b"oggbytes"


@patch("src.transcribe.OpenAI")
def test_whisper_raises_on_empty_audio(mock_openai_cls):
    import pytest
    with pytest.raises(ValueError, match="empty audio"):
        transcribe.whisper(b"", api_key="sk-1")
