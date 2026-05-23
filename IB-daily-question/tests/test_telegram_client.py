from unittest.mock import patch, Mock
from src import telegram_client as tg


def _mock_response(json_data, status=200):
    r = Mock()
    r.status_code = status
    r.json.return_value = json_data
    r.content = b"binary"
    r.raise_for_status = Mock()
    return r


@patch("src.telegram_client.requests.post")
def test_send_message_posts_to_api(mock_post):
    mock_post.return_value = _mock_response({"ok": True, "result": {"message_id": 42}})
    msg_id = tg.send_message("TOKEN", "CHAT", "Hello *world*")
    assert msg_id == 42
    args, kwargs = mock_post.call_args
    assert "bot TOKEN" not in args[0]  # token in URL path, not bearer
    assert "TOKEN" in args[0]
    assert kwargs["json"]["chat_id"] == "CHAT"
    assert kwargs["json"]["text"] == "Hello *world*"
    assert kwargs["json"]["parse_mode"] == "Markdown"


@patch("src.telegram_client.requests.get")
def test_get_updates_returns_results(mock_get):
    mock_get.return_value = _mock_response({"ok": True, "result": [{"update_id": 1}]})
    updates = tg.get_updates("TOKEN", offset=10, timeout=30)
    assert updates == [{"update_id": 1}]
    args, kwargs = mock_get.call_args
    assert kwargs["params"]["offset"] == 10
    assert kwargs["params"]["timeout"] == 30


@patch("src.telegram_client.requests.get")
def test_download_voice_two_step(mock_get):
    mock_get.side_effect = [
        _mock_response({"ok": True, "result": {"file_path": "voice/file_42.ogg"}}),
        _mock_response({}, status=200),
    ]
    audio = tg.download_voice("TOKEN", "FILE_ID")
    assert audio == b"binary"
    # first call: getFile
    assert "getFile" in mock_get.call_args_list[0][0][0]
    # second call: file download URL with file_path
    assert "voice/file_42.ogg" in mock_get.call_args_list[1][0][0]


@patch("src.telegram_client.requests.post")
def test_send_message_retries_on_5xx(mock_post):
    fail = _mock_response({}, status=500)
    fail.raise_for_status.side_effect = Exception("boom")
    ok = _mock_response({"ok": True, "result": {"message_id": 1}})
    mock_post.side_effect = [fail, fail, ok]
    msg_id = tg.send_message("TOKEN", "CHAT", "hi", max_retries=3, backoff=0)
    assert msg_id == 1
    assert mock_post.call_count == 3
