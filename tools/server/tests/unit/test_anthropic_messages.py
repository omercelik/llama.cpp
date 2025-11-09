import pytest

from tools.server.tests.utils import ServerPreset

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def _basic_payload(stream: bool = False):
    return {
        "model": server.model_alias,
        "max_tokens": 8,
        "system": "You are a test assistant.",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Say hi"},
                ],
            },
        ],
        "stream": stream,
    }


def test_anthropic_messages_basic():
    global server
    server.start()
    res = server.make_request("POST", "/v1/messages", data=_basic_payload())
    assert res.status_code == 200
    body = res.body
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["id"].startswith("msg_")
    assert "usage" in body
    assert body["usage"]["input_tokens"] > 0
    assert body["usage"]["output_tokens"] >= 0
    assert isinstance(body["content"], list)
    assert any(block.get("type") == "text" for block in body["content"])


def test_anthropic_messages_stream():
    global server
    server.start()
    stream = server.make_stream_request("POST", "/v1/messages", data=_basic_payload(stream=True))

    collected_text = []
    seen_start = False
    seen_stop = False
    for event in stream:
        event_type = event.get("event") or event.get("type")
        if event_type == "message_start":
            seen_start = True
            assert event["message"]["id"].startswith("msg_")
        elif event_type == "content_block_delta" and event.get("delta", {}).get("type") == "text_delta":
            collected_text.append(event["delta"]["text"])
        elif event_type == "message_delta":
            usage = event.get("usage", {})
            assert "output_tokens" in usage
            assert "input_tokens" in usage
        elif event_type == "message_stop":
            seen_stop = True

    assert seen_start
    assert seen_stop
    assert any(text.strip() for text in collected_text)
