import httpx
import pytest

from hypothesa import config
from hypothesa.healthcheck import _check_telegram, _migration_head


def test_migration_head_is_discoverable() -> None:
    assert _migration_head() == "20260710_0005"


def test_telegram_failure_does_not_expose_token(monkeypatch) -> None:
    token = "secret-token-must-not-leak"
    monkeypatch.setattr(config, "TELEGRAM_TOKEN", token)

    def fail(*args, **kwargs):
        request = httpx.Request("GET", f"https://api.telegram.org/bot{token}/getMe")
        raise httpx.RequestError("failed", request=request)

    monkeypatch.setattr(httpx, "get", fail)

    with pytest.raises(RuntimeError) as error:
        _check_telegram()

    assert token not in str(error.value)
