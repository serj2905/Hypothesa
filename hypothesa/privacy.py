"""Детерминированная минимизация PII перед ML-аналитикой."""

from __future__ import annotations

import re

_EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+", re.IGNORECASE)
_PHONE = re.compile(
    r"(?<!\d)(?:\+?7|8)[\s()\-]*(?:\d[\s()\-]*){10}(?!\d)"
)
_PASSPORT = re.compile(r"(?<!\d)\d{4}[\s-]?\d{6}(?!\d)")
_LONG_NUMBER = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")


def redact_pii(value: str) -> str:
    """Скрыть контактные и идентификационные данные, сохранив смысл ответа."""
    value = _EMAIL.sub("[EMAIL]", value)
    value = _PHONE.sub("[PHONE]", value)
    value = _PASSPORT.sub("[PASSPORT]", value)
    value = _LONG_NUMBER.sub("[LONG_NUMBER]", value)
    return " ".join(value.split())
