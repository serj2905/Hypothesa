"""REST API Hypothesa поверх общей доменной логики."""

from __future__ import annotations

import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Literal
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator

from . import config
from .experiments import assign_variant, pseudonymize_participant
from .interview import InterviewSession, advance, start_interview
from .llm import LLMClient
from .storage import ConcurrentSessionUpdate, Storage

logger = logging.getLogger(__name__)

InterviewState = Literal["awaiting_consent", "active", "completed", "declined"]


class StartInterviewRequest(BaseModel):
    participant_token: UUID = Field(
        description="Случайный UUID REST-клиента. Исходное значение в БД не сохраняется."
    )


class MessageRequest(BaseModel):
    participant_token: UUID
    message: str = Field(min_length=1, max_length=4000)

    @field_validator("message")
    @classmethod
    def message_must_contain_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Сообщение не должно быть пустым.")
        return value


class InterviewResponse(BaseModel):
    interview_id: UUID
    message: str
    state: InterviewState
    storage_status: str
    answered_questions: int
    total_questions: int
    variant: Literal["control", "adaptive"]
    survey_version: int


class DeleteParticipantResponse(BaseModel):
    deleted: bool


class HealthResponse(BaseModel):
    status: Literal["ok"]
    database: Literal["ok"]


class ServiceInfoResponse(BaseModel):
    service: Literal["Hypothesa API"]
    version: Literal["1.0.0"]
    docs: Literal["/api/docs"]
    respondent_interface: Literal["Telegram bot"]


def get_storage(request: Request) -> Storage:
    return request.app.state.storage


def get_generator(request: Request) -> LLMClient:
    return request.app.state.generator


StorageDependency = Annotated[Storage, Depends(get_storage)]
GeneratorDependency = Annotated[LLMClient, Depends(get_generator)]


def _participant_id(token: UUID) -> int:
    return pseudonymize_participant(token.hex, config.PARTICIPANT_SALT)


def _session_state(session: InterviewSession) -> InterviewState:
    if session.declined:
        return "declined"
    if session.finished:
        return "completed"
    if not session.consent_given:
        return "awaiting_consent"
    return "active"


def _resume_message(session: InterviewSession) -> str:
    if session.declined:
        return "Участие в опросе отменено."
    if session.finished:
        return "Спасибо! Интервью завершено, ответы сохранены."
    if not session.consent_given:
        return "Чтобы начать опрос, ответьте «Согласен» или «Не согласен»."
    question = session.current_question()
    pending = next(
        (turn for turn in reversed(question.followups) if turn.answer is None),
        None,
    )
    return pending.question if pending else question.spec.text


def _response(
    session: InterviewSession,
    message: str,
    *,
    storage_status: str,
) -> InterviewResponse:
    return InterviewResponse(
        interview_id=session.interview_id,
        message=message,
        state=_session_state(session),
        storage_status=storage_status,
        answered_questions=sum(question.answer is not None for question in session.questions),
        total_questions=len(session.questions),
        variant=session.variant,
        survey_version=session.survey_version,
    )


def create_app(
    *,
    storage: Storage | None = None,
    generator: LLMClient | None = None,
) -> FastAPI:
    """Собрать приложение; явные зависимости используются в unit-тестах."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        config.validate()
        owns_storage = storage is None
        owns_generator = generator is None
        app.state.storage = storage or Storage()
        app.state.generator = generator or LLMClient(model=config.LLM_MODEL)
        app.state.llm_slots = threading.BoundedSemaphore(config.API_MAX_CONCURRENT_LLM_REQUESTS)
        try:
            yield
        finally:
            if owns_generator:
                app.state.generator.close()
            if owns_storage:
                app.state.storage.close()

    app = FastAPI(
        title="Hypothesa API",
        summary="API адаптивного AI-интервьюера",
        description=(
            "Запускает интервью, принимает ответы респондента и передаёт их "
            "в общую доменную модель Hypothesa."
        ),
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )

    @app.get(
        "/",
        response_model=ServiceInfoResponse,
        tags=["Служебные"],
        summary="Получить информацию о сервисе",
    )
    def service_info() -> ServiceInfoResponse:
        return ServiceInfoResponse(
            service="Hypothesa API",
            version="1.0.0",
            docs="/api/docs",
            respondent_interface="Telegram bot",
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Служебные"],
        summary="Проверить API и соединение с БД",
    )
    def healthcheck(storage: StorageDependency) -> HealthResponse:
        try:
            storage.healthcheck()
        except Exception as exc:
            logger.warning("Healthcheck БД завершился ошибкой.", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="База данных недоступна.",
            ) from exc
        return HealthResponse(status="ok", database="ok")

    @app.post(
        "/api/v1/interviews",
        response_model=InterviewResponse,
        status_code=status.HTTP_201_CREATED,
        tags=["Интервью"],
        summary="Начать новое интервью",
    )
    def create_interview(
        payload: StartInterviewRequest,
        storage: StorageDependency,
    ) -> InterviewResponse:
        user_id = _participant_id(payload.participant_token)
        questionnaire = storage.load_latest_questionnaire(config.SURVEY_ID)
        variant = (
            assign_variant(
                user_id,
                config.SURVEY_ID,
                adaptive_share=config.ADAPTIVE_SHARE,
            )
            if questionnaire is not None
            else "control"
        )
        selected_questions = questionnaire.questions if variant == "adaptive" else None
        greeting, session = start_interview(
            selected_questions,
            survey_id=config.SURVEY_ID,
            survey_version=questionnaire.version if selected_questions else 1,
            variant=variant,
        )
        storage.start_session(user_id, session, replace_active=True)
        return _response(session, greeting, storage_status="active")

    @app.get(
        "/api/v1/interviews/{interview_id}",
        response_model=InterviewResponse,
        tags=["Интервью"],
        summary="Получить состояние интервью",
    )
    def get_interview(
        interview_id: UUID,
        storage: StorageDependency,
        participant_token: Annotated[UUID, Query()],
    ) -> InterviewResponse:
        stored = storage.load_session(
            _participant_id(participant_token),
            interview_id,
        )
        if stored is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Интервью не найдено.",
            )
        session, storage_status = stored
        return _response(
            session,
            _resume_message(session),
            storage_status=storage_status,
        )

    @app.post(
        "/api/v1/interviews/{interview_id}/messages",
        response_model=InterviewResponse,
        tags=["Интервью"],
        summary="Отправить ответ респондента",
    )
    def send_message(
        interview_id: UUID,
        payload: MessageRequest,
        request: Request,
        storage: StorageDependency,
        generator: GeneratorDependency,
    ) -> InterviewResponse:
        user_id = _participant_id(payload.participant_token)
        session = storage.load_active_session(user_id, config.SURVEY_ID)
        if session is None or session.interview_id != interview_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Активное интервью не найдено.",
            )

        try:
            with request.app.state.llm_slots:
                reply, session = advance(session, payload.message, generator)
            if session.declined:
                storage.delete_session(user_id, session.interview_id)
                storage_status = "deleted"
            else:
                storage.save_session(user_id, session)
                storage_status = "pending_summary" if session.finished else "active"
        except ConcurrentSessionUpdate as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Ответ уже обрабатывается. Повторите запрос чуть позже.",
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Не удалось обработать сообщение interview_id=%s", interview_id)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Модель временно недоступна. Попробуйте ещё раз.",
            ) from exc

        return _response(session, reply, storage_status=storage_status)

    @app.delete(
        "/api/v1/participants/me",
        response_model=DeleteParticipantResponse,
        tags=["Приватность"],
        summary="Удалить данные текущего респондента",
    )
    def delete_participant(
        storage: StorageDependency,
        participant_token: Annotated[UUID, Query()],
    ) -> DeleteParticipantResponse:
        deleted = storage.delete_participant(_participant_id(participant_token))
        return DeleteParticipantResponse(deleted=bool(deleted))

    return app


app = create_app()
