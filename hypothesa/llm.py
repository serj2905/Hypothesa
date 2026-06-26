"""Адаптер инференса — тонкая обёртка над Ollama HTTP API.

Провайдер-агностичный интерфейс: остальной код зовёт .structured(schema, messages)
и получает валидный экземпляр Pydantic-схемы. Гарантия формата — через параметр
Ollama `format` (JSON-схема передаётся декодеру), а не «попросили модель».

Если позже понадобится OpenAI/vLLM — меняется только этот файл.
"""

from __future__ import annotations

from typing import TypeVar

import httpx
from pydantic import BaseModel

from . import config

T = TypeVar("T", bound=BaseModel)

Message = dict[str, str]  # {"role": "system|user|assistant", "content": "..."}


class LLMClient:
    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ) -> None:
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        self.model = model or config.LLM_MODEL
        self.timeout = timeout or config.LLM_TIMEOUT

    def structured(
        self,
        schema: type[T],
        messages: list[Message],
        temperature: float = 0.0,
    ) -> T:
        """Вернуть ответ модели, провалидированный против Pydantic-схемы.

        temperature=0.0 по умолчанию — суммаризация и судьи должны быть
        детерминированы настолько, насколько возможно.
        """
        resp = httpx.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "format": schema.model_json_schema(),
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        return schema.model_validate_json(content)
