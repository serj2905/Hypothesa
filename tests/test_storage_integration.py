from __future__ import annotations

import os
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import delete, func, select

from hypothesa.interview import QuestionSpec, start_interview
from hypothesa.schemas import OpenAnswer
from hypothesa.storage import (
    ConcurrentSessionUpdate,
    Storage,
    completed_interviews,
    interview_sessions,
)
from hypothesa.summarize import CompletedInterview, OpenAnswerResult

pytestmark = pytest.mark.integration


@pytest.fixture()
def storage():
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL не задан")
    instance = Storage(url)
    instance.create_schema()
    yield instance
    instance.close()


def test_optimistic_lock_and_idempotent_finalize(storage: Storage) -> None:
    user_id = -int(datetime.now(UTC).timestamp() * 1_000_000)
    _, session = start_interview(
        [
            QuestionSpec(id=1, kind="open", text="Вопрос", max_followups=0),
            QuestionSpec(
                id=2,
                kind="open",
                text="Адаптивный вопрос",
                max_followups=0,
                topic_id=uuid4(),
            ),
        ]
    )
    stale = session.model_copy(deep=True)

    try:
        storage.start_session(user_id, session)
        session.questions[0].answer = "Ответ"
        session.questions[1].answer = "Навязанный темой ответ"
        session.current_index = 2
        session.finished = True
        session.completed_at = datetime.now(UTC)
        storage.save_session(user_id, session)

        with pytest.raises(ConcurrentSessionUpdate):
            storage.save_session(user_id, stale)

        claimed = storage.claim_finished_sessions(limit=10)
        claimed_session = next(item for uid, item in claimed if uid == user_id)
        completed = CompletedInterview(
            age=None,
            city=None,
            open_answers={
                1: OpenAnswerResult(
                    raw_answer="Ответ",
                    summary=OpenAnswer(items=["Ответ"]),
                    faithful=True,
                ),
                2: OpenAnswerResult(
                    raw_answer="Навязанный темой ответ",
                    summary=OpenAnswer(items=["Навязанный темой ответ"]),
                    faithful=True,
                ),
            },
        )
        storage.finalize_summary(user_id, claimed_session, completed)
        storage.finalize_summary(user_id, claimed_session, completed)

        refresh = storage.topic_refresh_state(session.survey_id)
        assert refresh.total_valid >= 1
        assert refresh.new_valid >= 1
        documents = [
            document
            for document in storage.list_topic_documents(session.survey_id)
            if document.interview_id == session.interview_id
        ]
        assert [document.question_id for document in documents] == [1]

        with (
            storage.advisory_lock("integration-lock") as first_lock,
            storage.advisory_lock("integration-lock") as second_lock,
        ):
            assert first_lock
            assert not second_lock

        with storage.engine.connect() as conn:
            count = conn.scalar(
                select(func.count())
                .select_from(completed_interviews)
                .where(completed_interviews.c.interview_id == session.interview_id)
            )
        assert count == 1

        # Завершённый запуск не мешает создать следующий и не перезаписывается.
        _, next_session = start_interview()
        storage.start_session(user_id, next_session)
        assert storage.load_active_session(user_id).interview_id == next_session.interview_id
        assert storage.delete_session(user_id, next_session.interview_id)
        assert storage.load_active_session(user_id) is None

        _, final_session = start_interview()
        storage.start_session(user_id, final_session)
        assert storage.delete_participant(user_id) == 2
        assert storage.load_active_session(user_id) is None
    finally:
        with storage.engine.begin() as conn:
            conn.execute(delete(completed_interviews).where(completed_interviews.c.user_id == user_id))
            conn.execute(delete(interview_sessions).where(interview_sessions.c.user_id == user_id))
