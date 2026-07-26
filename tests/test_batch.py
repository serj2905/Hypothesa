from datetime import UTC, datetime

from hypothesa.batch import run_pending_summaries
from hypothesa.interview import InterviewSession, QuestionSpec, QuestionState
from hypothesa.schemas import FaithfulnessVerdict, OpenAnswer


def completed_session() -> InterviewSession:
    now = datetime.now(UTC)
    return InterviewSession(
        questions=[
            QuestionState(
                spec=QuestionSpec(id=1, kind="age", text="Возраст?"),
                answer="30",
            ),
            QuestionState(
                spec=QuestionSpec(id=2, kind="city", text="Город?"),
                answer="Москва",
            ),
            QuestionState(
                spec=QuestionSpec(id=3, kind="open", text="Минусы?"),
                answer="Высокая комиссия",
            ),
            QuestionState(
                spec=QuestionSpec(id=4, kind="open", text="Плюсы?"),
                answer="Удобное приложение",
            ),
        ],
        current_index=4,
        finished=True,
        completed_at=now,
    )


class FakeStorage:
    def __init__(self, session):
        self.session = session
        self.finalized = []
        self.released = []

    def claim_finished_sessions(self, limit=50):
        return [(42, self.session)]

    def finalize_summary(self, user_id, session, result):
        self.finalized.append((user_id, session, result))

    def release_summary(self, interview_id, error):
        self.released.append((interview_id, error))


class Generator:
    def __init__(self, calls):
        self.calls = calls

    def structured(self, schema, messages, temperature=0.0):
        assert schema is OpenAnswer
        self.calls.append("generate")
        return OpenAnswer(items=[messages[-1]["content"]])

    def unload(self):
        self.calls.append("unload")


class Judge:
    def __init__(self, calls):
        self.calls = calls

    def structured(self, schema, messages, temperature=0.0):
        assert schema is FaithfulnessVerdict
        self.calls.append("judge")
        return FaithfulnessVerdict(faithful=True)


def test_batch_separates_generator_and_judge_phases() -> None:
    calls = []
    storage = FakeStorage(completed_session())

    result = run_pending_summaries(storage, Generator(calls), Judge(calls))

    assert calls == ["generate", "generate", "unload", "judge", "judge"]
    assert result.processed == [storage.session.interview_id]
    assert not result.failed
    assert len(storage.finalized) == 1
    assert not storage.released
