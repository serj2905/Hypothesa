"""Проверка конфигурации, миграций PostgreSQL и обеих Ollama-моделей."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import httpx
from alembic.config import Config as AlembicConfig
from alembic.script import ScriptDirectory
from sqlalchemy import text

from . import config
from .llm import LLMClient
from .storage import Storage


def _migration_head() -> str:
    project_root = Path(__file__).resolve().parents[1]
    alembic_config = AlembicConfig(str(project_root / "alembic.ini"))
    return ScriptDirectory.from_config(alembic_config).get_current_head()


def _check_migrations(storage: Storage) -> str:
    expected = _migration_head()
    with storage.engine.connect() as connection:
        current = connection.scalar(text("SELECT version_num FROM alembic_version"))
    if current != expected:
        raise RuntimeError(f"Версия БД {current!r}, ожидается Alembic head {expected!r}.")
    return str(current)


def _check_telegram() -> str:
    if not config.TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN не задан.")
    try:
        response = httpx.get(
            f"https://api.telegram.org/bot{config.TELEGRAM_TOKEN}/getMe",
            timeout=10,
        )
        if response.status_code != 200 or not response.json().get("ok"):
            raise RuntimeError(f"Telegram API вернул HTTP {response.status_code}.")
        username = response.json().get("result", {}).get("username", "unknown")
        return f"OK, bot=@{username}"
    except httpx.HTTPError as exc:
        # URL содержит токен, поэтому repr исходной ошибки выводить нельзя.
        raise RuntimeError(f"Telegram API недоступен: {type(exc).__name__}.") from None


def run_checks() -> list[dict[str, str | bool]]:
    """Выполнить все проверки, сохраняя ошибки без раскрытия секретов."""
    results: list[dict[str, str | bool]] = []

    config_errors = config.validation_errors(require_bot_secrets=True)
    results.append(
        {
            "check": "configuration",
            "ok": not config_errors,
            "detail": "OK" if not config_errors else "; ".join(config_errors),
        }
    )

    try:
        telegram_detail = _check_telegram()
        results.append({"check": "telegram", "ok": True, "detail": telegram_detail})
    except Exception as exc:
        results.append({"check": "telegram", "ok": False, "detail": repr(exc)})

    storage = Storage()
    try:
        storage.healthcheck()
        revision = _check_migrations(storage)
        results.append(
            {"check": "postgresql", "ok": True, "detail": f"OK, revision={revision}"}
        )
    except Exception as exc:
        results.append({"check": "postgresql", "ok": False, "detail": repr(exc)})
    finally:
        storage.close()

    for role, model in (("generator", config.LLM_MODEL), ("judge", config.JUDGE_MODEL)):
        try:
            with LLMClient(model=model, timeout=10, retry_count=0) as client:
                client.healthcheck()
            results.append(
                {"check": f"ollama_{role}", "ok": True, "detail": f"OK, model={model}"}
            )
        except Exception as exc:
            results.append(
                {"check": f"ollama_{role}", "ok": False, "detail": repr(exc)}
            )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true", help="Вывести результат в JSON.")
    args = parser.parse_args(argv)
    results = run_checks()
    if args.json:
        print(json.dumps(results, ensure_ascii=False))
    else:
        for result in results:
            status = "OK" if result["ok"] else "FAIL"
            print(f"{result['check']}: {status} — {result['detail']}")
    return 1 if any(not result["ok"] for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
