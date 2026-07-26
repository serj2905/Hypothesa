import json

import httpx
import pytest

from hypothesa.llm import LLMClient
from hypothesa.schemas import AgeAnswer


def test_structured_retries_server_error_and_reuses_schema() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, request=request)
        payload = {
            "message": {"content": json.dumps({"age": 30})},
            "total_duration": 1000,
            "load_duration": 100,
            "prompt_eval_count": 12,
            "prompt_eval_duration": 200,
            "eval_count": 3,
            "eval_duration": 300,
        }
        return httpx.Response(200, json=payload, request=request)

    client = LLMClient(
        host="http://test",
        retry_count=1,
        transport=httpx.MockTransport(handler),
    )

    assert client.structured(AgeAnswer, [{"role": "user", "content": "30"}]).age == 30
    assert calls == 2
    assert client.last_metrics is not None
    assert client.last_metrics.prompt_tokens == 12
    assert client.last_metrics.generated_tokens == 3
    client.close()


def test_structured_does_not_retry_client_error() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, request=request)

    client = LLMClient(
        host="http://test",
        retry_count=2,
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(httpx.HTTPStatusError):
        client.structured(AgeAnswer, [])
    assert calls == 1
    client.close()
