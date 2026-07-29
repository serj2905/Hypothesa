"""Провайдер-агностичный structured-output клиент для Ollama."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TypeVar

import httpx
from pydantic import BaseModel

from . import config

T = TypeVar("T", bound=BaseModel)
Message = dict[str, str]
logger = logging.getLogger(__name__)


def release_llm_client(client: object, role: str, *, close: bool) -> None:
    """Best-effort выгрузить модель и при необходимости закрыть HTTP-клиент."""
    actions = [("выгрузить модель", getattr(client, "unload", None))]
    if close:
        actions.append(("закрыть клиент", getattr(client, "close", None)))

    for description, action in actions:
        if not callable(action):
            continue
        try:
            action()
        except Exception:
            logger.warning(
                "Не удалось %s для роли %s.",
                description,
                role,
                exc_info=True,
            )


@dataclass(frozen=True)
class LLMCallMetrics:
    total_duration_ns: int | None = None
    load_duration_ns: int | None = None
    prompt_tokens: int | None = None
    prompt_duration_ns: int | None = None
    generated_tokens: int | None = None
    generation_duration_ns: int | None = None


class LLMClient:
    """Переиспользует HTTP-соединения и повторяет только временные ошибки."""

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
        retry_count: int | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.host = (host or config.OLLAMA_HOST).rstrip("/")
        self.model = model or config.LLM_MODEL
        self.timeout = timeout or config.LLM_TIMEOUT
        self.retry_count = config.LLM_RETRIES if retry_count is None else retry_count
        self.last_metrics: LLMCallMetrics | None = None
        self._client = httpx.Client(
            base_url=self.host,
            timeout=self.timeout,
            transport=transport,
        )

    def structured(
        self,
        schema: type[T],
        messages: list[Message],
        temperature: float = 0.0,
    ) -> T:
        payload = {
            "model": self.model,
            "messages": messages,
            "format": schema.model_json_schema(),
            "stream": False,
            "options": {"temperature": temperature},
        }
        last_error: Exception | None = None
        for attempt in range(self.retry_count + 1):
            try:
                response = self._client.post("/api/chat", json=payload)
                response.raise_for_status()
                body = response.json()
                self.last_metrics = LLMCallMetrics(
                    total_duration_ns=body.get("total_duration"),
                    load_duration_ns=body.get("load_duration"),
                    prompt_tokens=body.get("prompt_eval_count"),
                    prompt_duration_ns=body.get("prompt_eval_duration"),
                    generated_tokens=body.get("eval_count"),
                    generation_duration_ns=body.get("eval_duration"),
                )
                logger.info(
                    "llm_call model=%s total_ns=%s load_ns=%s prompt_tokens=%s generated_tokens=%s",
                    self.model,
                    self.last_metrics.total_duration_ns,
                    self.last_metrics.load_duration_ns,
                    self.last_metrics.prompt_tokens,
                    self.last_metrics.generated_tokens,
                )
                content = body["message"]["content"]
                return schema.model_validate_json(content)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500:
                    raise
                last_error = exc
            except (httpx.RequestError, KeyError, ValueError) as exc:
                last_error = exc

            if attempt < self.retry_count:
                time.sleep(min(0.25 * 2**attempt, 1.0))

        if last_error is None:
            raise RuntimeError("LLM-запрос завершился без результата и диагностической ошибки.")
        raise last_error

    def list_models(self) -> set[str]:
        response = self._client.get("/api/tags")
        response.raise_for_status()
        return {item["name"] for item in response.json().get("models", [])}

    def healthcheck(self, *, require_model: bool = True) -> None:
        models = self.list_models()
        if require_model and self.model not in models:
            raise RuntimeError(f"Модель {self.model!r} не загружена в Ollama.")

    def unload(self) -> None:
        response = self._client.post(
            "/api/generate",
            json={"model": self.model, "keep_alive": 0},
        )
        response.raise_for_status()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
