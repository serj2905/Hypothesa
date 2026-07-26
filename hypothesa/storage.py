"""PostgreSQL-хранилище интервью с optimistic locking и идемпотентным batch."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    Uuid,
    case,
    create_engine,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import IntegrityError

from . import config
from .interview import InterviewSession
from .privacy import redact_pii
from .summarize import CompletedInterview

STATUS_ACTIVE = "active"
STATUS_PENDING = "pending_summary"
STATUS_PROCESSING = "processing"
STATUS_SUMMARIZED = "summarized"
STATUS_ABANDONED = "abandoned"
STATUS_SUMMARY_FAILED = "summary_failed"

metadata = MetaData()
json_type = JSON().with_variant(JSONB, "postgresql")

interview_sessions = Table(
    "interview_sessions",
    metadata,
    Column("interview_id", Uuid(as_uuid=True), primary_key=True),
    Column("user_id", BigInteger, nullable=False),
    Column("survey_id", String(100), nullable=False),
    Column("survey_version", Integer, nullable=False),
    Column("variant", String(20), nullable=False),
    Column("session", json_type, nullable=False),
    Column("status", String(30), nullable=False),
    Column("revision", Integer, nullable=False, server_default="0"),
    Column("summary_attempts", Integer, nullable=False, server_default="0"),
    Column("last_error", Text),
    Column("processing_started_at", DateTime(timezone=True)),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)
Index("ix_interview_sessions_user_status", interview_sessions.c.user_id, interview_sessions.c.status)
Index("ix_interview_sessions_status_updated", interview_sessions.c.status, interview_sessions.c.updated_at)
Index(
    "uq_active_session_per_user_survey",
    interview_sessions.c.user_id,
    interview_sessions.c.survey_id,
    unique=True,
    postgresql_where=interview_sessions.c.status == STATUS_ACTIVE,
)

interview_events = Table(
    "interview_events",
    metadata,
    Column("event_id", Uuid(as_uuid=True), primary_key=True),
    Column(
        "interview_id",
        Uuid(as_uuid=True),
        ForeignKey("interview_sessions.interview_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("kind", String(30), nullable=False),
    Column("occurred_at", DateTime(timezone=True), nullable=False),
    Column("question_id", Integer),
    Column("details", json_type, nullable=False),
)
Index("ix_interview_events_interview_time", interview_events.c.interview_id, interview_events.c.occurred_at)

completed_interviews = Table(
    "completed_interviews",
    metadata,
    Column("interview_id", Uuid(as_uuid=True), primary_key=True),
    Column("user_id", BigInteger, nullable=False),
    Column("survey_id", String(100), nullable=False),
    Column("survey_version", Integer, nullable=False),
    Column("variant", String(20), nullable=False),
    Column("age", Integer),
    Column("city", Text),
    Column("open_answers", json_type, nullable=False),
    Column("faithful", Boolean, nullable=False),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True), nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)
Index("ix_completed_survey_created", completed_interviews.c.survey_id, completed_interviews.c.created_at)

topics = Table(
    "topics",
    metadata,
    Column("topic_id", Uuid(as_uuid=True), primary_key=True),
    Column("survey_id", String(100), nullable=False),
    Column("label", Text, nullable=False),
    Column("keywords", json_type, nullable=False),
    Column("centroid", json_type, nullable=False),
    Column("rating", Integer, nullable=False),
    Column("mention_count", Integer, nullable=False),
    Column("active", Boolean, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)
Index("ix_topics_survey_active_rating", topics.c.survey_id, topics.c.active, topics.c.rating)

topic_assignments = Table(
    "topic_assignments",
    metadata,
    Column("document_id", Uuid(as_uuid=True), primary_key=True),
    Column("survey_id", String(100), nullable=False),
    Column("interview_id", Uuid(as_uuid=True), nullable=False),
    Column("question_id", Integer, nullable=False),
    Column("item_index", Integer, nullable=False),
    Column("text", Text, nullable=False),
    Column("topic_id", Uuid(as_uuid=True), ForeignKey("topics.topic_id")),
    Column("probability", Float),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)
Index("ix_topic_assignments_survey_topic", topic_assignments.c.survey_id, topic_assignments.c.topic_id)

questionnaire_versions = Table(
    "questionnaire_versions",
    metadata,
    Column("survey_id", String(100), primary_key=True),
    Column("version", Integer, primary_key=True),
    Column("questions", json_type, nullable=False),
    Column("source_topic_ids", json_type, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

topic_runs = Table(
    "topic_runs",
    metadata,
    Column("run_id", Uuid(as_uuid=True), primary_key=True),
    Column("survey_id", String(100), nullable=False),
    Column("questionnaire_version", Integer, nullable=False),
    Column("document_count", Integer, nullable=False),
    Column("metrics", json_type, nullable=False),
    Column("data_cutoff", DateTime(timezone=True)),
    Column("data_cutoff_created_at", DateTime(timezone=True)),
    Column("started_at", DateTime(timezone=True), nullable=False),
    Column("completed_at", DateTime(timezone=True), nullable=False),
)
Index("ix_topic_runs_survey_completed", topic_runs.c.survey_id, topic_runs.c.completed_at)


class StorageError(RuntimeError):
    """Базовая ошибка слоя хранения."""


class ActiveSessionExists(StorageError):
    """У пользователя уже есть активное интервью этого исследования."""


class ConcurrentSessionUpdate(StorageError):
    """Сессию успел изменить другой обработчик."""


@dataclass(frozen=True)
class TopicRefreshState:
    total_valid: int
    new_valid: int
    last_run_at: datetime | None
    last_data_cutoff: datetime | None


class Storage:
    """Тонкая синхронная обёртка; подключение открывается только при операции."""

    def __init__(self, database_url: str | None = None) -> None:
        self.engine = create_engine(
            database_url or config.DATABASE_URL,
            pool_pre_ping=True,
        )

    def create_schema(self) -> None:
        """Только для тестов/локального старта; production использует Alembic."""
        metadata.create_all(self.engine)

    def close(self) -> None:
        self.engine.dispose()

    def healthcheck(self) -> None:
        with self.engine.connect() as conn:
            conn.execute(select(1))

    @contextmanager
    def advisory_lock(self, key: str) -> Iterator[bool]:
        """Удерживать process-wide PostgreSQL lock на отдельном соединении."""
        raw = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
        lock_id = raw if raw < 2**63 else raw - 2**64
        connection = self.engine.connect()
        acquired = bool(
            connection.scalar(select(func.pg_try_advisory_lock(lock_id)))
        )
        try:
            yield acquired
        finally:
            if acquired:
                connection.execute(select(func.pg_advisory_unlock(lock_id)))
            connection.close()

    def active_session_count(
        self,
        survey_id: str,
        *,
        recent_within: timedelta | None = None,
    ) -> int:
        conditions = [
            interview_sessions.c.survey_id == survey_id,
            interview_sessions.c.status == STATUS_ACTIVE,
        ]
        if recent_within is not None:
            conditions.append(
                interview_sessions.c.updated_at
                >= datetime.now(UTC) - recent_within
            )
        stmt = (
            select(func.count())
            .select_from(interview_sessions)
            .where(*conditions)
        )
        with self.engine.connect() as conn:
            return int(conn.scalar(stmt) or 0)

    def topic_refresh_state(self, survey_id: str) -> TopicRefreshState:
        """Сколько faithful-интервью появилось после последнего topic run."""
        valid_condition = (
            completed_interviews.c.survey_id == survey_id,
            completed_interviews.c.faithful.is_(True),
        )
        last_run_stmt = (
            select(
                topic_runs.c.completed_at,
                topic_runs.c.data_cutoff_created_at,
            )
            .where(topic_runs.c.survey_id == survey_id)
            .order_by(topic_runs.c.completed_at.desc())
            .limit(1)
        )
        with self.engine.connect() as conn:
            total_valid = int(
                conn.scalar(
                    select(func.count())
                    .select_from(completed_interviews)
                    .where(*valid_condition)
                )
                or 0
            )
            last_run = conn.execute(last_run_stmt).first()
            cutoff = (
                last_run.data_cutoff_created_at or last_run.completed_at
                if last_run
                else None
            )
            if cutoff is None:
                new_valid = total_valid
            else:
                new_valid = int(
                    conn.scalar(
                        select(func.count())
                        .select_from(completed_interviews)
                        .where(
                            *valid_condition,
                            completed_interviews.c.created_at > cutoff,
                        )
                    )
                    or 0
                )
        return TopicRefreshState(
            total_valid=total_valid,
            new_valid=new_valid,
            last_run_at=last_run.completed_at if last_run else None,
            last_data_cutoff=cutoff,
        )

    def start_session(
        self,
        user_id: int,
        session: InterviewSession,
        *,
        replace_active: bool = False,
    ) -> None:
        """Создать новый запуск, не перезаписывая ожидающие batch-сессии."""
        with self.engine.begin() as conn:
            if replace_active:
                conn.execute(
                    update(interview_sessions)
                    .where(
                        interview_sessions.c.user_id == user_id,
                        interview_sessions.c.survey_id == session.survey_id,
                        interview_sessions.c.status == STATUS_ACTIVE,
                    )
                    .values(status=STATUS_ABANDONED, updated_at=func.now())
                )
            try:
                conn.execute(
                    insert(interview_sessions).values(
                        interview_id=session.interview_id,
                        user_id=user_id,
                        survey_id=session.survey_id,
                        survey_version=session.survey_version,
                        variant=session.variant,
                        session=session.model_dump(mode="json"),
                        status=STATUS_ACTIVE,
                        revision=session.revision,
                    )
                )
                self._insert_events(conn, session)
            except IntegrityError as exc:
                raise ActiveSessionExists(
                    "Активное интервью уже существует; используйте replace_active=True."
                ) from exc

    def load_active_session(self, user_id: int, survey_id: str | None = None) -> InterviewSession | None:
        conditions = [
            interview_sessions.c.user_id == user_id,
            interview_sessions.c.status == STATUS_ACTIVE,
        ]
        if survey_id is not None:
            conditions.append(interview_sessions.c.survey_id == survey_id)
        stmt = (
            select(interview_sessions.c.session)
            .where(*conditions)
            .order_by(interview_sessions.c.created_at.desc())
            .limit(1)
        )
        with self.engine.connect() as conn:
            row = conn.execute(stmt).first()
        return InterviewSession.model_validate(row.session) if row else None

    def save_session(self, user_id: int, session: InterviewSession) -> None:
        """Сохранить только ожидаемую ревизию, защищая ответы от lost update."""
        next_revision = session.revision + 1
        persisted = session.model_copy(update={"revision": next_revision})
        next_status = (
            STATUS_ABANDONED
            if session.declined
            else STATUS_PENDING
            if session.finished
            else STATUS_ACTIVE
        )
        stmt = (
            update(interview_sessions)
            .where(
                interview_sessions.c.interview_id == session.interview_id,
                interview_sessions.c.user_id == user_id,
                interview_sessions.c.status == STATUS_ACTIVE,
                interview_sessions.c.revision == session.revision,
            )
            .values(
                session=persisted.model_dump(mode="json"),
                status=next_status,
                revision=next_revision,
                updated_at=func.now(),
            )
        )
        with self.engine.begin() as conn:
            result = conn.execute(stmt)
            if result.rowcount != 1:
                raise ConcurrentSessionUpdate(
                    f"Сессия {session.interview_id} уже изменена или завершена."
                )
            self._insert_events(conn, persisted)
        session.revision = next_revision

    @staticmethod
    def _insert_events(conn, session: InterviewSession) -> None:
        for event in session.events:
            statement = pg_insert(interview_events).values(
                event_id=event.event_id,
                interview_id=session.interview_id,
                kind=event.kind,
                occurred_at=event.occurred_at,
                question_id=event.question_id,
                details=event.details,
            )
            conn.execute(
                statement.on_conflict_do_nothing(
                    index_elements=[interview_events.c.event_id]
                )
            )

    def claim_finished_sessions(
        self,
        *,
        limit: int = 50,
        stale_after: timedelta = timedelta(minutes=30),
        max_attempts: int = 3,
    ) -> list[tuple[int, InterviewSession]]:
        """Атомарно забрать batch-задачи через ``FOR UPDATE SKIP LOCKED``."""
        stale_before = datetime.now(UTC) - stale_after
        with self.engine.begin() as conn:
            conn.execute(
                update(interview_sessions)
                .where(
                    interview_sessions.c.status == STATUS_PROCESSING,
                    interview_sessions.c.processing_started_at < stale_before,
                )
                .values(status=STATUS_PENDING, processing_started_at=None, updated_at=func.now())
            )
            rows = conn.execute(
                select(
                    interview_sessions.c.interview_id,
                    interview_sessions.c.user_id,
                    interview_sessions.c.session,
                )
                .where(
                    interview_sessions.c.status == STATUS_PENDING,
                    interview_sessions.c.summary_attempts < max_attempts,
                )
                .order_by(interview_sessions.c.updated_at)
                .limit(limit)
                .with_for_update(skip_locked=True)
            ).all()
            ids = [row.interview_id for row in rows]
            if ids:
                conn.execute(
                    update(interview_sessions)
                    .where(interview_sessions.c.interview_id.in_(ids))
                    .values(
                        status=STATUS_PROCESSING,
                        processing_started_at=func.now(),
                        updated_at=func.now(),
                    )
                )
        return [
            (row.user_id, InterviewSession.model_validate(row.session)) for row in rows
        ]

    def finalize_summary(
        self,
        user_id: int,
        session: InterviewSession,
        result: CompletedInterview,
    ) -> None:
        """Идемпотентно записать результат и подтвердить задачу одной транзакцией."""
        if session.completed_at is None:
            raise StorageError("У завершённой сессии отсутствует completed_at.")
        dumped = result.model_dump(mode="json")
        faithful = all(answer["faithful"] for answer in dumped["open_answers"].values())
        completed = pg_insert(completed_interviews).values(
            interview_id=session.interview_id,
            user_id=user_id,
            survey_id=session.survey_id,
            survey_version=session.survey_version,
            variant=session.variant,
            age=dumped["age"],
            city=dumped["city"],
            open_answers=dumped["open_answers"],
            faithful=faithful,
            started_at=session.started_at,
            completed_at=session.completed_at,
        )
        completed = completed.on_conflict_do_nothing(
            index_elements=[completed_interviews.c.interview_id]
        )
        with self.engine.begin() as conn:
            conn.execute(completed)
            conn.execute(
                update(interview_sessions)
                .where(interview_sessions.c.interview_id == session.interview_id)
                .values(
                    status=STATUS_SUMMARIZED,
                    processing_started_at=None,
                    last_error=None,
                    updated_at=func.now(),
                )
            )

    def release_summary(
        self,
        interview_id: UUID,
        error: Exception,
        *,
        max_attempts: int = 3,
    ) -> None:
        """Вернуть неуспешную задачу в очередь и сохранить диагностический текст."""
        with self.engine.begin() as conn:
            conn.execute(
                update(interview_sessions)
                .where(
                    interview_sessions.c.interview_id == interview_id,
                    interview_sessions.c.status == STATUS_PROCESSING,
                )
                .values(
                    status=case(
                        (
                            interview_sessions.c.summary_attempts + 1 >= max_attempts,
                            STATUS_SUMMARY_FAILED,
                        ),
                        else_=STATUS_PENDING,
                    ),
                    summary_attempts=interview_sessions.c.summary_attempts + 1,
                    last_error=repr(error)[:4000],
                    processing_started_at=None,
                    updated_at=func.now(),
                )
            )

    def completed_count(self, survey_id: str | None = None) -> int:
        stmt = select(func.count()).select_from(completed_interviews)
        if survey_id is not None:
            stmt = stmt.where(completed_interviews.c.survey_id == survey_id)
        with self.engine.connect() as conn:
            return int(conn.scalar(stmt) or 0)

    def list_topic_documents(self, survey_id: str):
        """Развернуть базовые faithful items в обезличенный корпус BERTopic.

        Ответы на вопросы, которые уже были порождены темой (`topic_id != None`),
        намеренно исключаются: иначе адаптивная анкета сама усиливает выбранные темы.
        """
        from .topics import TopicDocument

        stmt = select(
            completed_interviews.c.interview_id,
            completed_interviews.c.open_answers,
            interview_sessions.c.session,
        ).select_from(
            completed_interviews.join(
                interview_sessions,
                completed_interviews.c.interview_id == interview_sessions.c.interview_id,
            )
        ).where(completed_interviews.c.survey_id == survey_id)
        documents = []
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).all()
        for row in rows:
            session = InterviewSession.model_validate(row.session)
            discovery_question_ids = {
                question.spec.id
                for question in session.questions
                if question.spec.kind == "open" and question.spec.topic_id is None
            }
            for question_id, answer in row.open_answers.items():
                if int(question_id) not in discovery_question_ids:
                    continue
                if not answer.get("faithful", False):
                    continue
                for item_index, text in enumerate(answer["summary"].get("items", [])):
                    safe_text = redact_pii(str(text))
                    normalized = safe_text.lower()
                    document_id = uuid5(
                        NAMESPACE_URL,
                        f"hypothesa:{row.interview_id}:{question_id}:{item_index}:{normalized}",
                    )
                    documents.append(
                        TopicDocument(
                            document_id=document_id,
                            interview_id=row.interview_id,
                            question_id=int(question_id),
                            item_index=item_index,
                            text=safe_text,
                        )
                    )
        return documents

    def load_topics(self, survey_id: str):
        from .topics import RegisteredTopic

        stmt = select(topics).where(topics.c.survey_id == survey_id)
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).mappings().all()
        return [
            RegisteredTopic(
                topic_id=row["topic_id"],
                survey_id=row["survey_id"],
                label=row["label"],
                keywords=row["keywords"],
                centroid=row["centroid"],
                rating=row["rating"],
                mention_count=row["mention_count"],
                active=row["active"],
            )
            for row in rows
        ]

    def next_questionnaire_version(self, survey_id: str) -> int:
        stmt = select(func.max(questionnaire_versions.c.version)).where(
            questionnaire_versions.c.survey_id == survey_id
        )
        with self.engine.connect() as conn:
            latest = conn.scalar(stmt)
        return int(latest or 0) + 1

    def save_topic_cycle(self, result) -> None:
        """Атомарно заменить assignments, обновить темы и добавить анкету."""
        documents = {
            document.document_id: document
            for document in self.list_topic_documents(result.questionnaire.survey_id)
        }
        with self.engine.begin() as conn:
            for topic in result.topics:
                statement = pg_insert(topics).values(
                    topic_id=topic.topic_id,
                    survey_id=topic.survey_id,
                    label=topic.label,
                    keywords=topic.keywords,
                    centroid=topic.centroid,
                    rating=topic.rating,
                    mention_count=topic.mention_count,
                    active=topic.active,
                )
                conn.execute(
                    statement.on_conflict_do_update(
                        index_elements=[topics.c.topic_id],
                        set_={
                            "label": statement.excluded.label,
                            "keywords": statement.excluded.keywords,
                            "centroid": statement.excluded.centroid,
                            "rating": statement.excluded.rating,
                            "mention_count": statement.excluded.mention_count,
                            "active": statement.excluded.active,
                            "updated_at": func.now(),
                        },
                    )
                )

            survey_id = result.questionnaire.survey_id
            conn.execute(
                topic_assignments.delete().where(
                    topic_assignments.c.survey_id == survey_id
                )
            )
            for assignment in result.assignments:
                document = documents[assignment.document_id]
                conn.execute(
                    insert(topic_assignments).values(
                        document_id=document.document_id,
                        survey_id=survey_id,
                        interview_id=document.interview_id,
                        question_id=document.question_id,
                        item_index=document.item_index,
                        text=document.text,
                        topic_id=assignment.topic_id,
                        probability=assignment.probability,
                    )
                )

            if result.questionnaire_changed:
                conn.execute(
                    insert(questionnaire_versions).values(
                        survey_id=survey_id,
                        version=result.questionnaire.version,
                        questions=[
                            question.model_dump(mode="json")
                            for question in result.questionnaire.questions
                        ],
                        source_topic_ids=[
                            str(topic_id)
                            for topic_id in result.questionnaire.source_topic_ids
                        ],
                    )
                )
            data_cutoff = conn.scalar(
                select(func.max(completed_interviews.c.completed_at)).where(
                    completed_interviews.c.survey_id == survey_id
                )
            )
            data_cutoff_created_at = conn.scalar(
                select(func.max(completed_interviews.c.created_at)).where(
                    completed_interviews.c.survey_id == survey_id
                )
            )
            metrics = dict(result.metrics)
            if data_cutoff is not None:
                metrics["time_to_insight_seconds"] = max(
                    0.0,
                    (result.completed_at - data_cutoff).total_seconds(),
                )
            conn.execute(
                insert(topic_runs).values(
                    run_id=result.run_id,
                    survey_id=survey_id,
                    questionnaire_version=result.questionnaire.version,
                    document_count=result.document_count,
                    metrics=metrics,
                    data_cutoff=data_cutoff,
                    data_cutoff_created_at=data_cutoff_created_at,
                    started_at=result.started_at,
                    completed_at=result.completed_at,
                )
            )

    def load_latest_questionnaire(self, survey_id: str):
        from .topics import QuestionnaireVersion

        stmt = (
            select(questionnaire_versions)
            .where(questionnaire_versions.c.survey_id == survey_id)
            .order_by(questionnaire_versions.c.version.desc())
            .limit(1)
        )
        with self.engine.connect() as conn:
            row = conn.execute(stmt).mappings().first()
        if row is None:
            return None
        from .interview import QuestionSpec

        return QuestionnaireVersion(
            survey_id=row["survey_id"],
            version=row["version"],
            questions=[QuestionSpec.model_validate(question) for question in row["questions"]],
            source_topic_ids=[UUID(value) for value in row["source_topic_ids"]],
        )

    def list_experiment_records(self, survey_id: str):
        from .metrics import ExperimentRecord

        # До первой adaptive-анкеты эксперимента ещё нет: исторический control
        # нельзя смешивать с одновременными control/adaptive группами.
        with self.engine.connect() as conn:
            experiment_started_at = conn.scalar(
                select(func.min(questionnaire_versions.c.created_at)).where(
                    questionnaire_versions.c.survey_id == survey_id
                )
            )
        if experiment_started_at is None:
            return []

        stmt = (
            select(
                interview_sessions.c.user_id,
                interview_sessions.c.session,
                interview_sessions.c.status,
                completed_interviews.c.faithful,
            )
            .select_from(
                interview_sessions.outerjoin(
                    completed_interviews,
                    interview_sessions.c.interview_id
                    == completed_interviews.c.interview_id,
                )
            )
            .where(
                interview_sessions.c.survey_id == survey_id,
                interview_sessions.c.created_at >= experiment_started_at,
            )
            .order_by(
                interview_sessions.c.user_id,
                interview_sessions.c.created_at,
            )
        )
        with self.engine.connect() as conn:
            rows = conn.execute(stmt).all()
        records: dict[int, ExperimentRecord] = {}
        for row in rows:
            records.setdefault(
                row.user_id,
                ExperimentRecord(
                    session=InterviewSession.model_validate(row.session),
                    status=row.status,
                    faithful=row.faithful,
                    participant_id=row.user_id,
                ),
            )
        return list(records.values())

    def delete_participant(self, user_id: int) -> int:
        """Удалить псевдонимизированные индивидуальные данные по запросу участника."""
        with self.engine.begin() as conn:
            interview_ids = list(
                conn.scalars(
                    select(interview_sessions.c.interview_id).where(
                        interview_sessions.c.user_id == user_id
                    )
                )
            )
            if interview_ids:
                conn.execute(
                    topic_assignments.delete().where(
                        topic_assignments.c.interview_id.in_(interview_ids)
                    )
                )
            conn.execute(
                completed_interviews.delete().where(
                    completed_interviews.c.user_id == user_id
                )
            )
            deleted = conn.execute(
                interview_sessions.delete().where(
                    interview_sessions.c.user_id == user_id
                )
            ).rowcount
        return int(deleted or 0)

    def delete_session(self, user_id: int, interview_id: UUID) -> bool:
        """Удалить один запуск, не затрагивая прошлые интервью участника."""
        with self.engine.begin() as conn:
            conn.execute(
                topic_assignments.delete().where(
                    topic_assignments.c.interview_id == interview_id
                )
            )
            conn.execute(
                completed_interviews.delete().where(
                    completed_interviews.c.interview_id == interview_id,
                    completed_interviews.c.user_id == user_id,
                )
            )
            deleted = conn.execute(
                interview_sessions.delete().where(
                    interview_sessions.c.interview_id == interview_id,
                    interview_sessions.c.user_id == user_id,
                )
            ).rowcount
        return bool(deleted)

    def purge_expired_data(self, retention_days: int) -> int:
        """Удалить индивидуальные записи старше retention; агрегированные темы остаются."""
        if retention_days < 1:
            raise ValueError("retention_days должен быть положительным.")
        cutoff = datetime.now(UTC) - timedelta(days=retention_days)
        with self.engine.begin() as conn:
            completed_ids = set(
                conn.scalars(
                    select(completed_interviews.c.interview_id).where(
                        completed_interviews.c.created_at < cutoff
                    )
                )
            )
            session_ids = set(
                conn.scalars(
                    select(interview_sessions.c.interview_id).where(
                        interview_sessions.c.created_at < cutoff
                    )
                )
            )
            expired_ids = completed_ids | session_ids
            if not expired_ids:
                return 0
            conn.execute(
                topic_assignments.delete().where(
                    topic_assignments.c.interview_id.in_(expired_ids)
                )
            )
            conn.execute(
                completed_interviews.delete().where(
                    completed_interviews.c.interview_id.in_(expired_ids)
                )
            )
            conn.execute(
                interview_sessions.delete().where(
                    interview_sessions.c.interview_id.in_(expired_ids)
                )
            )
        return len(expired_ids)
