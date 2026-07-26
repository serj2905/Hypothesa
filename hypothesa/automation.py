"""Безопасный автоматический batch → threshold → BERTopic pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import UUID

from . import config
from .batch import BatchResult, run_pending_summaries
from .llm import LLMClient
from .storage import Storage, TopicRefreshState
from .topics import BERTopicBackend, InsufficientDocuments, TopicBackend, run_topic_cycle

AutomationStatus = Literal[
    "locked",
    "active_sessions",
    "below_minimum",
    "waiting_for_new",
    "cooldown",
    "insufficient_documents",
    "topics_refreshed",
    "questionnaire_updated",
]


@dataclass(frozen=True)
class AutomationResult:
    status: AutomationStatus
    batch_processed: int = 0
    batch_failed: int = 0
    total_valid: int = 0
    new_valid: int = 0
    active_sessions: int = 0
    topic_run_id: UUID | None = None
    questionnaire_version: int | None = None
    questionnaire_changed: bool = False
    detail: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def _common_result_fields(
    batch: BatchResult,
    state: TopicRefreshState,
    active_sessions: int,
) -> dict:
    return {
        "batch_processed": len(batch.processed),
        "batch_failed": len(batch.failed),
        "total_valid": state.total_valid,
        "new_valid": state.new_valid,
        "active_sessions": active_sessions,
    }


def _process_pending_summaries(
    storage: Storage,
    generator: LLMClient | None,
    judge: LLMClient | None,
) -> BatchResult:
    # Один крупный batch сохраняет разделение VRAM-фаз: весь Qwen, затем весь Llama.
    # Остаток сверх лимита будет обработан на следующем scheduler tick.
    return run_pending_summaries(
        storage,
        generator,
        judge,
        limit=config.AUTOMATION_BATCH_LIMIT,
    )


def run_automatic_pipeline_once(
    storage: Storage,
    *,
    backend: TopicBackend | None = None,
    generator: LLMClient | None = None,
    judge: LLMClient | None = None,
    now: datetime | None = None,
) -> AutomationResult:
    """Выполнить один защищённый проход автоматического конвейера."""
    now = now or datetime.now(UTC)
    lock_key = f"hypothesa:automation:{config.SURVEY_ID}"
    with storage.advisory_lock(lock_key) as acquired:
        if not acquired:
            return AutomationResult(
                status="locked",
                detail="Другой scheduler уже обрабатывает это исследование.",
            )

        active_sessions = storage.active_session_count(
            config.SURVEY_ID,
            recent_within=timedelta(
                minutes=config.AUTOMATION_ACTIVE_GRACE_MINUTES
            ),
        )
        if config.AUTOMATION_REQUIRE_NO_ACTIVE_SESSIONS and active_sessions:
            return AutomationResult(
                status="active_sessions",
                active_sessions=active_sessions,
                detail="Есть активные интервью; offline-фаза отложена.",
            )

        batch = _process_pending_summaries(storage, generator, judge)
        state = storage.topic_refresh_state(config.SURVEY_ID)
        common = _common_result_fields(batch, state, active_sessions)

        if state.total_valid < config.TOPIC_MIN_VALID_INTERVIEWS:
            return AutomationResult(
                status="below_minimum",
                detail=(
                    f"Нужно {config.TOPIC_MIN_VALID_INTERVIEWS} faithful-интервью, "
                    f"сейчас {state.total_valid}."
                ),
                **common,
            )

        if state.last_run_at is not None and state.new_valid < config.TOPIC_REFRESH_EVERY:
            return AutomationResult(
                status="waiting_for_new",
                detail=(
                    f"Нужно {config.TOPIC_REFRESH_EVERY} новых faithful-интервью, "
                    f"сейчас {state.new_valid}."
                ),
                **common,
            )

        cooldown = timedelta(hours=config.TOPIC_COOLDOWN_HOURS)
        if state.last_run_at is not None and now - state.last_run_at < cooldown:
            ready_at = state.last_run_at + cooldown
            return AutomationResult(
                status="cooldown",
                detail=f"Следующий тематический запуск разрешён после {ready_at.isoformat()}.",
                **common,
            )

        backend = backend or BERTopicBackend(
            embedding_model=config.EMBEDDING_MODEL,
            random_state=config.TOPIC_RANDOM_STATE,
            min_topic_size=config.TOPIC_MIN_SIZE,
            device=config.EMBEDDING_DEVICE,
        )
        try:
            topic_result = run_topic_cycle(
                storage,
                config.SURVEY_ID,
                backend,
                similarity_threshold=config.TOPIC_SIMILARITY_THRESHOLD,
                min_mentions=config.TOPIC_MIN_MENTIONS,
                min_prevalence=config.TOPIC_MIN_PREVALENCE,
                topic_question_count=config.TOPIC_QUESTION_COUNT,
                seed=config.TOPIC_RANDOM_STATE,
            )
        except InsufficientDocuments as exc:
            return AutomationResult(
                status="insufficient_documents",
                detail=str(exc),
                **common,
            )

        return AutomationResult(
            status=(
                "questionnaire_updated"
                if topic_result.questionnaire_changed
                else "topics_refreshed"
            ),
            topic_run_id=topic_result.run_id,
            questionnaire_version=topic_result.questionnaire.version,
            questionnaire_changed=topic_result.questionnaire_changed,
            detail=(
                "Создана новая адаптивная анкета."
                if topic_result.questionnaire_changed
                else "Темы пересчитаны; состав анкеты не изменился."
            ),
            **common,
        )
