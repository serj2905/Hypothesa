"""Продуктовые метрики интервью и A/B-гейт без внешней BI-системы."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import mean, median

from .interview import InterviewSession


@dataclass(frozen=True)
class ExperimentRecord:
    session: InterviewSession
    status: str
    faithful: bool | None
    participant_id: int | None = None


@dataclass(frozen=True)
class VariantMetrics:
    started: int
    completed: int
    valid_completed: int
    completion_rate: float
    valid_completion_rate: float
    completion_ci95: tuple[float, float]
    median_duration_seconds: float | None
    average_followups: float
    followup_session_rate: float
    followup_answer_rate: float
    average_initial_answer_characters: float
    average_followup_answer_characters: float
    average_question_coverage: float


@dataclass(frozen=True)
class ExperimentReport:
    survey_id: str
    variants: dict[str, VariantMetrics]
    completion_rate_delta: float | None
    valid_completion_rate_delta: float | None
    ready_for_decision: bool
    readiness_reason: str

    def as_dict(self) -> dict:
        return asdict(self)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Доверительный интервал доли по Уилсону.

    В отличие от нормального приближения остаётся корректным на малых выборках
    и у границ 0/1, поэтому им пользуются и completion rate, и распространённость
    тем (topics.calculate_ratings).
    """
    if total == 0:
        return (0.0, 0.0)
    rate = successes / total
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _variant_metrics(records: list[ExperimentRecord]) -> VariantMetrics:
    started = len(records)
    completed_records = [record for record in records if record.session.finished]
    valid_completed = sum(record.faithful is True for record in records)
    durations = [
        (record.session.completed_at - record.session.started_at).total_seconds()
        for record in completed_records
        if record.session.completed_at is not None
    ]
    followups = [
        sum(event.kind == "followup_asked" for event in record.session.events) for record in records
    ]
    open_questions = [
        question
        for record in records
        for question in record.session.questions
        if question.spec.kind == "open"
    ]
    followup_turns = [turn for question in open_questions for turn in question.followups]
    answered_followups = [turn for turn in followup_turns if turn.answer is not None]
    initial_lengths = [
        len(question.initial_answer)
        for question in open_questions
        if question.initial_answer is not None
    ]
    followup_lengths = [len(turn.answer) for turn in answered_followups if turn.answer is not None]
    sessions_with_followup = sum(
        any(question.followups for question in record.session.questions) for record in records
    )
    coverages = []
    for record in records:
        completed_ids = {
            event.question_id
            for event in record.session.events
            if event.kind == "question_completed"
        }
        coverages.append(len(completed_ids) / len(record.session.questions))

    return VariantMetrics(
        started=started,
        completed=len(completed_records),
        valid_completed=valid_completed,
        completion_rate=len(completed_records) / started if started else 0.0,
        valid_completion_rate=valid_completed / started if started else 0.0,
        completion_ci95=wilson_interval(len(completed_records), started),
        median_duration_seconds=median(durations) if durations else None,
        average_followups=mean(followups) if followups else 0.0,
        followup_session_rate=(sessions_with_followup / started if started else 0.0),
        followup_answer_rate=(
            len(answered_followups) / len(followup_turns) if followup_turns else 0.0
        ),
        average_initial_answer_characters=(mean(initial_lengths) if initial_lengths else 0.0),
        average_followup_answer_characters=(mean(followup_lengths) if followup_lengths else 0.0),
        average_question_coverage=mean(coverages) if coverages else 0.0,
    )


def build_experiment_report(
    survey_id: str,
    records: list[ExperimentRecord],
    *,
    minimum_total: int = 50,
    minimum_per_variant: int = 20,
) -> ExperimentReport:
    # Повторные /start одного участника не должны увеличивать размер выборки.
    unique_records: list[ExperimentRecord] = []
    seen_participants: set[int] = set()
    for record in records:
        if record.participant_id is not None:
            if record.participant_id in seen_participants:
                continue
            seen_participants.add(record.participant_id)
        unique_records.append(record)
    records = unique_records
    grouped = {
        variant: [record for record in records if record.session.variant == variant]
        for variant in ("control", "adaptive")
    }
    variants = {name: _variant_metrics(group) for name, group in grouped.items()}
    control = variants["control"]
    adaptive = variants["adaptive"]
    both_present = control.started > 0 and adaptive.started > 0
    ready = (
        len(records) >= minimum_total
        and control.started >= minimum_per_variant
        and adaptive.started >= minimum_per_variant
    )
    if ready:
        reason = "Достигнут минимальный размер обеих групп; можно проводить продуктовый разбор."
    else:
        reason = (
            f"Нужно >={minimum_total} интервью и >={minimum_per_variant} в каждой группе; "
            f"сейчас total={len(records)}, control={control.started}, adaptive={adaptive.started}."
        )
    return ExperimentReport(
        survey_id=survey_id,
        variants=variants,
        completion_rate_delta=(
            adaptive.completion_rate - control.completion_rate if both_present else None
        ),
        valid_completion_rate_delta=(
            adaptive.valid_completion_rate - control.valid_completion_rate if both_present else None
        ),
        ready_for_decision=ready,
        readiness_reason=reason,
    )
