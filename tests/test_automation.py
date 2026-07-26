from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from hypothesa import config
from hypothesa.automation import run_automatic_pipeline_once
from hypothesa.storage import TopicRefreshState
from hypothesa.topics import (
    DiscoveredTopic,
    LocalAssignment,
    TopicDiscovery,
    TopicDocument,
)


class FakeBackend:
    def fit(self, documents):
        return TopicDiscovery(
            topics=[
                DiscoveredTopic(
                    local_id=0,
                    label="Комиссия",
                    keywords=["комиссия"],
                    centroid=[1.0, 0.0],
                )
            ],
            assignments=[LocalAssignment(item.document_id, 0, 0.9) for item in documents],
            metrics={"noise_ratio": 0.0},
        )


class FakeStorage:
    def __init__(self, state: TopicRefreshState, *, active=0, lock=True):
        interview_id = uuid4()
        self.state = state
        self.active = active
        self.lock = lock
        self.latest = None
        self.saved = None
        self.documents = [
            TopicDocument(uuid4(), interview_id, 3, 0, "Высокая комиссия")
        ]

    @contextmanager
    def advisory_lock(self, key):
        yield self.lock

    def active_session_count(self, survey_id, *, recent_within=None):
        return self.active

    def claim_finished_sessions(self, limit=50):
        return []

    def topic_refresh_state(self, survey_id):
        return self.state

    def list_topic_documents(self, survey_id):
        return self.documents

    def load_topics(self, survey_id):
        return []

    def next_questionnaire_version(self, survey_id):
        return 1 if self.latest is None else self.latest.version + 1

    def load_latest_questionnaire(self, survey_id):
        return self.latest

    def save_topic_cycle(self, result):
        self.saved = result
        if result.questionnaire_changed:
            self.latest = result.questionnaire


def configure(monkeypatch) -> None:
    monkeypatch.setattr(config, "TOPIC_MIN_VALID_INTERVIEWS", 50)
    monkeypatch.setattr(config, "TOPIC_REFRESH_EVERY", 10)
    monkeypatch.setattr(config, "TOPIC_COOLDOWN_HOURS", 24)
    monkeypatch.setattr(config, "TOPIC_MIN_MENTIONS", 1)
    monkeypatch.setattr(config, "TOPIC_MIN_PREVALENCE", 0.05)
    monkeypatch.setattr(config, "AUTOMATION_REQUIRE_NO_ACTIVE_SESSIONS", True)


def test_automation_waits_for_minimum_valid_interviews(monkeypatch) -> None:
    configure(monkeypatch)
    storage = FakeStorage(TopicRefreshState(49, 49, None, None))

    result = run_automatic_pipeline_once(storage, backend=FakeBackend())

    assert result.status == "below_minimum"
    assert storage.saved is None


def test_automation_defers_while_interview_is_active(monkeypatch) -> None:
    configure(monkeypatch)
    storage = FakeStorage(TopicRefreshState(50, 50, None, None), active=1)

    result = run_automatic_pipeline_once(storage, backend=FakeBackend())

    assert result.status == "active_sessions"


def test_automation_respects_cooldown(monkeypatch) -> None:
    configure(monkeypatch)
    now = datetime.now(UTC)
    storage = FakeStorage(
        TopicRefreshState(60, 10, now - timedelta(hours=2), now - timedelta(hours=2))
    )

    result = run_automatic_pipeline_once(storage, backend=FakeBackend(), now=now)

    assert result.status == "cooldown"


def test_automation_runs_topic_cycle_after_threshold(monkeypatch) -> None:
    configure(monkeypatch)
    storage = FakeStorage(TopicRefreshState(50, 50, None, None))

    result = run_automatic_pipeline_once(storage, backend=FakeBackend())

    assert result.status == "questionnaire_updated"
    assert result.questionnaire_changed
    assert storage.saved is not None
