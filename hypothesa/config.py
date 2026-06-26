"""Конфиг из окружения. Секреты и хосты — только через .env, не в коде.

Заменяет хардкод токенов в DHA_hackathon/sets.py.
"""

from __future__ import annotations

import os

# Подхватываем .env до чтения переменных. Мягкий импорт: в средах, где python-dotenv
# не установлен (а конфиг берётся из реального окружения), модуль не должен падать.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass


def _get(name: str, default: str) -> str:
    return os.getenv(name, default)


# Ollama HTTP-эндпоинт. На 5080-боксе — localhost.
OLLAMA_HOST: str = _get("OLLAMA_HOST", "http://localhost:11434")

# Генератор (интервьюер + суммаризатор) — Qwen, live во время опроса.
LLM_MODEL: str = _get("LLM_MODEL", "qwen2.5:14b-instruct-q4_K_M")

# Судья — ДРУГОЕ семейство (см. sprint_1.md): иначе self-preference bias.
# Запускается в batch-фазе, после выгрузки генератора из VRAM.
JUDGE_MODEL: str = _get("JUDGE_MODEL", "llama3.1:8b")

# Таймаут одного запроса к модели, сек.
LLM_TIMEOUT: int = int(_get("LLM_TIMEOUT", "120"))
