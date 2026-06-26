"""Pydantic-схемы ответов — контракт, который убивает классы багов
нестабильного формата суммаризации (баги 1 и 3 из CLAUDE.md).

Вместо «попросили LLM вернуть Python-список и парсим regex'ом» —
жёсткая схема, которую модель обязана заполнить через structured output.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgeAnswer(BaseModel):
    """Вопрос 1 — возраст."""

    age: int | None = Field(
        default=None,
        description="Возраст респондента в полных годах. null, если в ответе нет возраста.",
        ge=0,
        le=120,
    )


class CityAnswer(BaseModel):
    """Вопрос 2 — город."""

    city: str | None = Field(
        default=None,
        description="Нормализованное название города. null, если города в ответе нет.",
    )


class OpenAnswer(BaseModel):
    """Вопросы 3 и 4 — открытые ответы (негатив / позитив).

    Каждый элемент items — один отдельный аспект, дословно близко к ответу
    респондента, без домысливания. Пустой список, если конкретики нет
    («всё нравится», «ничего», «нет»).
    """

    items: list[str] = Field(
        default_factory=list,
        description=(
            "Список отдельных аспектов/проблем, явно упомянутых респондентом. "
            "Не добавляй ничего, чего нет в исходном ответе. "
            "Пустой список, если конкретных аспектов не названо."
        ),
    )


class FaithfulnessVerdict(BaseModel):
    """Вердикт Judge 2 — есть ли в суммаризации информация, которой нет в ответе."""

    faithful: bool = Field(
        description="true, если КАЖДЫЙ элемент суммаризации подтверждается исходным ответом."
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Элементы суммаризации, которых нет в исходном ответе (галлюцинации).",
    )
