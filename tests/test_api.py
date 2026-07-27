from __future__ import annotations

from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from hypothesa.api import create_app
from hypothesa.schemas import AgeAnswer


class FakeStorage:
    def __init__(self) -> None:
        self.records: dict[tuple[int, UUID], tuple[object, str]] = {}

    def healthcheck(self) -> None:
        return None

    def load_latest_questionnaire(self, _survey_id):
        return None

    def start_session(self, user_id, session, *, replace_active=False) -> None:
        if replace_active:
            for key, (stored, state) in list(self.records.items()):
                if key[0] == user_id and state == "active":
                    self.records[key] = (stored, "abandoned")
        self.records[(user_id, session.interview_id)] = (session, "active")

    def load_active_session(self, user_id, survey_id=None):
        for (owner_id, _), (session, state) in reversed(list(self.records.items())):
            if (
                owner_id == user_id
                and state == "active"
                and (survey_id is None or session.survey_id == survey_id)
            ):
                return session
        return None

    def load_session(self, user_id, interview_id):
        return self.records.get((user_id, interview_id))

    def save_session(self, user_id, session) -> None:
        session.revision += 1
        state = "pending_summary" if session.finished else "active"
        self.records[(user_id, session.interview_id)] = (session, state)

    def delete_session(self, user_id, interview_id) -> bool:
        return self.records.pop((user_id, interview_id), None) is not None

    def delete_participant(self, user_id) -> int:
        keys = [key for key in self.records if key[0] == user_id]
        for key in keys:
            self.records.pop(key)
        return len(keys)


class StubGenerator:
    def structured(self, schema, *_args, **_kwargs):
        if schema is AgeAnswer:
            return AgeAnswer(age=31)
        raise AssertionError(f"Неожиданный вызов модели со схемой {schema}.")


def test_rest_interview_flow() -> None:
    storage = FakeStorage()
    token = uuid4()

    with TestClient(create_app(storage=storage, generator=StubGenerator())) as client:
        index = client.get("/")
        assert index.status_code == 200
        assert index.json() == {
            "service": "Hypothesa API",
            "version": "1.0.0",
            "docs": "/api/docs",
            "respondent_interface": "Telegram bot",
        }

        health = client.get("/health")
        assert health.json() == {"status": "ok", "database": "ok"}

        started = client.post(
            "/api/v1/interviews",
            json={"participant_token": str(token)},
        )
        assert started.status_code == 201
        payload = started.json()
        interview_id = payload["interview_id"]
        assert payload["state"] == "awaiting_consent"
        assert payload["variant"] == "control"

        consent = client.post(
            f"/api/v1/interviews/{interview_id}/messages",
            json={
                "participant_token": str(token),
                "message": "Согласен",
            },
        )
        assert consent.status_code == 200
        assert consent.json()["state"] == "active"
        assert "лет" in consent.json()["message"].lower()

        age = client.post(
            f"/api/v1/interviews/{interview_id}/messages",
            json={
                "participant_token": str(token),
                "message": "Мне 31 год",
            },
        )
        assert age.status_code == 200
        assert "города" in age.json()["message"].lower()
        assert age.json()["answered_questions"] == 1

        restored = client.get(
            f"/api/v1/interviews/{interview_id}",
            params={"participant_token": str(token)},
        )
        assert restored.status_code == 200
        assert restored.json()["message"] == age.json()["message"]

        deleted = client.delete(
            "/api/v1/participants/me",
            params={"participant_token": str(token)},
        )
        assert deleted.json() == {"deleted": True}


def test_api_rejects_invalid_or_empty_messages() -> None:
    storage = FakeStorage()
    token = uuid4()

    with TestClient(create_app(storage=storage, generator=StubGenerator())) as client:
        started = client.post(
            "/api/v1/interviews",
            json={"participant_token": str(token)},
        ).json()

        response = client.post(
            f"/api/v1/interviews/{started['interview_id']}/messages",
            json={"participant_token": str(token), "message": "   "},
        )
        assert response.status_code == 422

        чужой_опрос = client.get(
            f"/api/v1/interviews/{started['interview_id']}",
            params={"participant_token": str(uuid4())},
        )
        assert чужой_опрос.status_code == 404
