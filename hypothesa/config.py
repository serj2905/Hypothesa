"""Конфиг из окружения. Секреты и хосты — только через .env, не в коде."""

from __future__ import annotations

import os
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# Подхватываем .env до чтения переменных. Мягкий импорт: в средах, где python-dotenv
# не установлен (а конфиг берётся из реального окружения), модуль не должен падать.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:
    pass


def _get(name: str, default: str) -> str:
    return os.getenv(name, default)


def _get_bool(name: str, default: bool) -> bool:
    value = _get(name, str(default)).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} должен быть true/false, получено {value!r}.")


# Ollama HTTP-эндпоинт. На 5080-боксе — localhost.
OLLAMA_HOST: str = _get("OLLAMA_HOST", "http://localhost:11434")

# Генератор (интервьюер + суммаризатор) — Qwen, live во время опроса.
LLM_MODEL: str = _get("LLM_MODEL", "qwen2.5:14b-instruct-q4_K_M")

# Судья — ДРУГОЕ семейство: иначе возникает self-preference bias.
# Запускается в batch-фазе, после выгрузки генератора из VRAM.
JUDGE_MODEL: str = _get("JUDGE_MODEL", "llama3.1:8b")

# Таймаут одного запроса к модели, сек.
LLM_TIMEOUT: int = int(_get("LLM_TIMEOUT", "120"))
LLM_RETRIES: int = int(_get("LLM_RETRIES", "2"))

# Postgres — состояние интервью и завершённые записи (см. storage.py).
# Заменяет joblib-файлы в dialogs/ прототипа: снимает гонки при параллельных
# диалогах и даёт единую точку для будущей аналитики (BERTopic, рейтинг).
DATABASE_URL: str = _get(
    "DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/hypothesa"
)

# Telegram-бот (bot.py). Без дефолта — пусть падает явно при старте, если токен
# не задан, а не тихо.
TELEGRAM_TOKEN: str = _get("TELEGRAM_TOKEN", "")

# Эксперимент и приватность. В production PARTICIPANT_SALT обязателен и хранится
# только в secret manager/.env; Telegram user id в БД не записывается.
SURVEY_ID: str = _get("SURVEY_ID", "sber-service-quality")
PARTICIPANT_SALT: str = _get("PARTICIPANT_SALT", "dev-only-change-me")
ADAPTIVE_SHARE: float = float(_get("ADAPTIVE_SHARE", "0.5"))
DATA_RETENTION_DAYS: int = int(_get("DATA_RETENTION_DAYS", "90"))
BOT_MAX_CONCURRENT_LLM_REQUESTS: int = int(
    _get("BOT_MAX_CONCURRENT_LLM_REQUESTS", "1")
)
API_MAX_CONCURRENT_LLM_REQUESTS: int = int(
    _get("API_MAX_CONCURRENT_LLM_REQUESTS", "2")
)
WORKER_BATCH_LIMIT: int = int(_get("WORKER_BATCH_LIMIT", "10"))
WORKER_POLL_SECONDS: float = float(_get("WORKER_POLL_SECONDS", "5"))

# Тематический цикл.
EMBEDDING_MODEL: str = _get(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
EMBEDDING_DEVICE: str = _get("EMBEDDING_DEVICE", "cpu")
TOPIC_RANDOM_STATE: int = int(_get("TOPIC_RANDOM_STATE", "42"))
TOPIC_MIN_SIZE: int = int(_get("TOPIC_MIN_SIZE", "5"))
TOPIC_SIMILARITY_THRESHOLD: float = float(_get("TOPIC_SIMILARITY_THRESHOLD", "0.82"))
TOPIC_MIN_MENTIONS: int = int(_get("TOPIC_MIN_MENTIONS", "2"))
TOPIC_MIN_PREVALENCE: float = float(_get("TOPIC_MIN_PREVALENCE", "0.05"))
TOPIC_QUESTION_COUNT: int = int(_get("TOPIC_QUESTION_COUNT", "2"))

# Автоматический offline-конвейер. Запускается отдельным scheduler-процессом,
# а не из Telegram handler, чтобы не блокировать живой диалог.
TOPIC_MIN_VALID_INTERVIEWS: int = int(_get("TOPIC_MIN_VALID_INTERVIEWS", "50"))
TOPIC_REFRESH_EVERY: int = int(_get("TOPIC_REFRESH_EVERY", "10"))
TOPIC_COOLDOWN_HOURS: int = int(_get("TOPIC_COOLDOWN_HOURS", "24"))
AUTOMATION_INTERVAL_MINUTES: int = int(_get("AUTOMATION_INTERVAL_MINUTES", "30"))
AUTOMATION_BATCH_LIMIT: int = int(_get("AUTOMATION_BATCH_LIMIT", "500"))
AUTOMATION_TIMEZONE: str = _get("AUTOMATION_TIMEZONE", "Europe/Moscow")
AUTOMATION_WINDOW_START_HOUR: int = int(_get("AUTOMATION_WINDOW_START_HOUR", "2"))
AUTOMATION_WINDOW_HOURS: int = int(_get("AUTOMATION_WINDOW_HOURS", "3"))
AUTOMATION_REQUIRE_NO_ACTIVE_SESSIONS: bool = _get_bool(
    "AUTOMATION_REQUIRE_NO_ACTIVE_SESSIONS", True
)
AUTOMATION_ACTIVE_GRACE_MINUTES: int = int(
    _get("AUTOMATION_ACTIVE_GRACE_MINUTES", "30")
)


class ConfigurationError(ValueError):
    """Окружение содержит небезопасные или противоречивые настройки."""


def validation_errors(*, require_bot_secrets: bool = False) -> list[str]:
    """Вернуть все ошибки конфигурации за один проход, не раскрывая секреты."""
    errors: list[str] = []
    if require_bot_secrets and not TELEGRAM_TOKEN:
        errors.append("TELEGRAM_TOKEN не задан.")
    if require_bot_secrets and PARTICIPANT_SALT in {
        "dev-only-change-me",
        "replace-with-a-long-random-secret",
    }:
        errors.append("PARTICIPANT_SALT должен быть заменён.")
    if len(PARTICIPANT_SALT) < 16:
        errors.append("PARTICIPANT_SALT должен содержать не меньше 16 символов.")
    if not 0.0 <= ADAPTIVE_SHARE <= 1.0:
        errors.append("ADAPTIVE_SHARE должен находиться между 0 и 1.")
    if not SURVEY_ID.strip():
        errors.append("SURVEY_ID не должен быть пустым.")
    for name, value in (
        ("LLM_TIMEOUT", LLM_TIMEOUT),
        ("DATA_RETENTION_DAYS", DATA_RETENTION_DAYS),
        ("BOT_MAX_CONCURRENT_LLM_REQUESTS", BOT_MAX_CONCURRENT_LLM_REQUESTS),
        ("API_MAX_CONCURRENT_LLM_REQUESTS", API_MAX_CONCURRENT_LLM_REQUESTS),
        ("WORKER_BATCH_LIMIT", WORKER_BATCH_LIMIT),
        ("TOPIC_MIN_SIZE", TOPIC_MIN_SIZE),
        ("TOPIC_MIN_MENTIONS", TOPIC_MIN_MENTIONS),
        ("TOPIC_MIN_VALID_INTERVIEWS", TOPIC_MIN_VALID_INTERVIEWS),
        ("TOPIC_REFRESH_EVERY", TOPIC_REFRESH_EVERY),
        ("AUTOMATION_INTERVAL_MINUTES", AUTOMATION_INTERVAL_MINUTES),
        ("AUTOMATION_BATCH_LIMIT", AUTOMATION_BATCH_LIMIT),
        ("AUTOMATION_ACTIVE_GRACE_MINUTES", AUTOMATION_ACTIVE_GRACE_MINUTES),
    ):
        if value < 1:
            errors.append(f"{name} должен быть положительным.")
    if WORKER_POLL_SECONDS <= 0:
        errors.append("WORKER_POLL_SECONDS должен быть положительным.")
    if LLM_RETRIES < 0:
        errors.append("LLM_RETRIES не должен быть отрицательным.")
    if not 0 <= TOPIC_SIMILARITY_THRESHOLD <= 1:
        errors.append("TOPIC_SIMILARITY_THRESHOLD должен находиться между 0 и 1.")
    if not 0 <= TOPIC_MIN_PREVALENCE <= 1:
        errors.append("TOPIC_MIN_PREVALENCE должен находиться между 0 и 1.")
    if not 0 <= AUTOMATION_WINDOW_START_HOUR <= 23:
        errors.append("AUTOMATION_WINDOW_START_HOUR должен быть от 0 до 23.")
    if not 1 <= AUTOMATION_WINDOW_HOURS <= 24:
        errors.append("AUTOMATION_WINDOW_HOURS должен быть от 1 до 24.")
    try:
        ZoneInfo(AUTOMATION_TIMEZONE)
    except ZoneInfoNotFoundError:
        errors.append(f"Неизвестный часовой пояс AUTOMATION_TIMEZONE={AUTOMATION_TIMEZONE!r}.")
    return errors


def validate(*, require_bot_secrets: bool = False) -> None:
    errors = validation_errors(require_bot_secrets=require_bot_secrets)
    if errors:
        raise ConfigurationError("Ошибки конфигурации:\n- " + "\n- ".join(errors))
