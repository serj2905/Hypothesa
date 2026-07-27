"""Детерминированное A/B-распределение без хранения Telegram user id."""

from __future__ import annotations

import hashlib
import hmac
from typing import Literal

ExperimentVariant = Literal["control", "adaptive"]


def pseudonymize_participant(raw_user_id: int | str, salt: str) -> int:
    """Получить стабильный внутренний id, не сохраняя внешний идентификатор."""
    if not salt:
        raise ValueError("Для псевдонимизации требуется непустой salt.")
    digest = hmac.new(
        salt.encode("utf-8"),
        str(raw_user_id).encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def assign_variant(
    participant_id: int,
    survey_id: str,
    *,
    adaptive_share: float = 0.5,
) -> ExperimentVariant:
    if not 0.0 <= adaptive_share <= 1.0:
        raise ValueError("adaptive_share должен находиться между 0 и 1.")
    digest = hashlib.sha256(f"{survey_id}:{participant_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / (1 << 64)
    return "adaptive" if bucket < adaptive_share else "control"
