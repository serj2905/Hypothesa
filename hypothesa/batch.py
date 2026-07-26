"""Идемпотентная batch-суммаризация с раздельными фазами generator/judge."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from uuid import UUID

from . import config
from .interview import InterviewSession
from .llm import LLMClient
from .storage import Storage
from .summarize import DraftInterview, generate_session_draft, judge_session_draft

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    processed: list[UUID] = field(default_factory=list)
    failed: dict[UUID, str] = field(default_factory=dict)


def _best_effort_unload(client: object, role: str) -> None:
    unload = getattr(client, "unload", None)
    if callable(unload):
        try:
            unload()
        except Exception:
            logger.warning("Не удалось выгрузить %s из Ollama.", role, exc_info=True)


def run_pending_summaries(
    storage: Storage,
    generator: LLMClient | None = None,
    judge: LLMClient | None = None,
    *,
    limit: int = 50,
) -> BatchResult:
    """Обработать очередь без чередования моделей в VRAM.

    Сначала Qwen строит drafts для всего batch, затем выгружается; только после
    этого Llama проверяет все drafts. Неуспешные задачи возвращаются в очередь.
    """
    result = BatchResult()
    claimed = storage.claim_finished_sessions(limit=limit)
    if not claimed:
        return result

    owns_generator = generator is None
    generator = generator or LLMClient(model=config.LLM_MODEL)
    drafts: list[tuple[int, InterviewSession, DraftInterview]] = []
    for user_id, session in claimed:
        try:
            drafts.append((user_id, session, generate_session_draft(session, generator)))
        except Exception as exc:  # одна плохая запись не останавливает batch
            storage.release_summary(session.interview_id, exc)
            result.failed[session.interview_id] = repr(exc)

    _best_effort_unload(generator, "generator")
    if owns_generator:
        generator.close()

    if not drafts:
        return result

    owns_judge = judge is None
    judge = judge or LLMClient(model=config.JUDGE_MODEL)
    for user_id, session, draft in drafts:
        try:
            completed = judge_session_draft(draft, judge)
            storage.finalize_summary(user_id, session, completed)
            result.processed.append(session.interview_id)
        except Exception as exc:
            storage.release_summary(session.interview_id, exc)
            result.failed[session.interview_id] = repr(exc)
    _best_effort_unload(judge, "judge")
    if owns_judge:
        judge.close()
    return result
