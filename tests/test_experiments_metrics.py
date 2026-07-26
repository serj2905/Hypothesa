from dataclasses import replace
from datetime import UTC, datetime, timedelta

from hypothesa.experiments import assign_variant, pseudonymize_participant
from hypothesa.interview import QuestionSpec, advance, start_interview
from hypothesa.metrics import ExperimentRecord, build_experiment_report


class NoFollowupLLM:
    def structured(self, schema, messages, temperature=0.0):
        from hypothesa.schemas import InterviewTurnDecision

        assert schema is InterviewTurnDecision
        return InterviewTurnDecision(has_more_to_ask=False)


def finished_record(variant: str, faithful: bool = True) -> ExperimentRecord:
    _, session = start_interview(
        [QuestionSpec(id=1, kind="open", text="Вопрос", max_followups=1)],
        variant=variant,
    )
    _, session = advance(session, "Согласен", NoFollowupLLM())
    _, session = advance(session, "Ответ", NoFollowupLLM())
    session.started_at = datetime.now(UTC) - timedelta(seconds=30)
    session.completed_at = session.started_at + timedelta(seconds=20)
    return ExperimentRecord(session=session, status="summarized", faithful=faithful)


def test_pseudonymization_and_assignment_are_stable() -> None:
    first = pseudonymize_participant(123, "secret")
    second = pseudonymize_participant(123, "secret")

    assert first == second
    assert first != 123
    assert assign_variant(first, "survey") == assign_variant(first, "survey")


def test_report_calculates_ab_delta_and_readiness_gate() -> None:
    records = [finished_record("control"), finished_record("adaptive")]

    report = build_experiment_report(
        "survey", records, minimum_total=2, minimum_per_variant=1
    )

    assert report.ready_for_decision
    assert report.completion_rate_delta == 0.0
    assert report.variants["control"].completion_rate == 1.0
    assert report.variants["adaptive"].median_duration_seconds == 20.0


def test_report_counts_participant_only_once() -> None:
    first = finished_record("control")
    duplicate = finished_record("control")
    adaptive = finished_record("adaptive")
    first = replace(first, participant_id=10)
    duplicate = replace(duplicate, participant_id=10)
    adaptive = replace(adaptive, participant_id=20)

    report = build_experiment_report(
        "survey",
        [first, duplicate, adaptive],
        minimum_total=2,
        minimum_per_variant=1,
    )

    assert report.variants["control"].started == 1
    assert report.variants["adaptive"].started == 1
    assert report.ready_for_decision
