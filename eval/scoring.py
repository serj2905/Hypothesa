"""Детерминированная оценка суммаризаций на вручную размеченных кейсах."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from statistics import mean

from eval.golden_cases import GoldenCase
from hypothesa.schemas import FaithfulnessVerdict, OpenAnswer


@dataclass(frozen=True)
class CaseScore:
    name: str
    format_valid: bool
    faithful: bool | None
    passed: bool
    items: list[str]
    concept_recall: float | None = None
    concept_precision: float | None = None
    has_banned: bool = False
    error: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def normalize_text(value: str) -> str:
    """Нормализовать регистр, ``ё`` и пунктуацию для лексической сверки."""
    value = unicodedata.normalize("NFKC", value).lower().replace("ё", "е")
    value = re.sub(r"[^\w\s:.]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def concept_recall(case: GoldenCase, items: Iterable[str]) -> float | None:
    """Доля эталонных понятий, найденных хотя бы в одной суммаризации."""
    if not case.expected_concepts:
        return None

    text = normalize_text(" ".join(items))
    hits = 0
    for alternatives in case.expected_concepts:
        normalized = (normalize_text(value) for value in alternatives)
        if any(value and value in text for value in normalized):
            hits += 1
    return hits / len(case.expected_concepts)


def concept_precision(case: GoldenCase, items: Iterable[str]) -> float | None:
    """Доля извлечённых items, связанных хотя бы с одним эталонным понятием."""
    if not case.expected_concepts:
        return None
    normalized_concepts = [
        normalize_text(value) for alternatives in case.expected_concepts for value in alternatives
    ]
    normalized_items = [normalize_text(item) for item in items]
    if not normalized_items:
        return 0.0
    supported = sum(
        any(concept and concept in item for concept in normalized_concepts)
        for item in normalized_items
    )
    return supported / len(normalized_items)


def score_case(
    case: GoldenCase,
    summary: OpenAnswer,
    verdict: FaithfulnessVerdict,
) -> CaseScore:
    """Оценить один валидный structured-output ответ."""
    normalized = normalize_text(" ".join(summary.items))
    has_banned = any(normalize_text(value) in normalized for value in case.banned_substrings)
    recall = concept_recall(case, summary.items)
    precision = concept_precision(case, summary.items)

    if case.expect_empty:
        passed = not summary.items
    else:
        passed = bool(
            summary.items
            and verdict.faithful
            and not has_banned
            and recall is not None
            and recall >= case.min_concept_recall
            and precision is not None
            and precision >= case.min_concept_precision
        )

    return CaseScore(
        name=case.name,
        format_valid=True,
        faithful=verdict.faithful,
        passed=passed,
        items=summary.items,
        concept_recall=recall,
        concept_precision=precision,
        has_banned=has_banned,
    )


def failed_case(case: GoldenCase, error: Exception) -> CaseScore:
    return CaseScore(
        name=case.name,
        format_valid=False,
        faithful=None,
        passed=False,
        items=[],
        concept_recall=0.0 if case.expected_concepts else None,
        concept_precision=0.0 if case.expected_concepts else None,
        error=repr(error),
    )


def aggregate_scores(scores: list[CaseScore]) -> dict[str, float]:
    """Посчитать метрики без деления на ноль и смешения разных типов кейсов."""
    if not scores:
        raise ValueError("Нельзя агрегировать пустой список eval-результатов.")

    empty_scores = [score for score in scores if score.concept_recall is None]
    content_scores = [score for score in scores if score.concept_recall is not None]
    recalls = [score.concept_recall for score in content_scores]
    precisions = [score.concept_precision for score in content_scores]

    return {
        "format_valid_rate": mean(score.format_valid for score in scores),
        "passed_rate": mean(score.passed for score in scores),
        "faithfulness_rate": mean(score.faithful is True for score in scores),
        "hallucination_rate": mean(score.has_banned or score.faithful is False for score in scores),
        "empty_correct_rate": mean(score.passed for score in empty_scores) if empty_scores else 1.0,
        "content_non_empty_rate": mean(bool(score.items) for score in content_scores)
        if content_scores
        else 1.0,
        "mean_concept_recall": mean(recalls) if recalls else 1.0,
        "mean_concept_precision": mean(precisions) if precisions else 1.0,
    }
