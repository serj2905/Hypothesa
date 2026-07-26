from hypothesa.interview import QuestionSpec, advance, start_interview
from hypothesa.schemas import AgeAnswer, CityAnswer, InterviewTurnDecision


class InterviewLLM:
    def structured(self, schema, messages, temperature=0.0):
        if schema is AgeAnswer:
            return AgeAnswer(age=30)
        if schema is CityAnswer:
            return CityAnswer(city="Москва")
        if schema is InterviewTurnDecision:
            return InterviewTurnDecision(has_more_to_ask=False)
        raise AssertionError(f"Неожиданная схема: {schema}")


class OneFollowupLLM(InterviewLLM):
    def structured(self, schema, messages, temperature=0.0):
        if schema is InterviewTurnDecision:
            return InterviewTurnDecision(
                has_more_to_ask=True,
                followup_question="Можете привести пример?",
                followup_reason="missing_context",
            )
        return super().structured(schema, messages, temperature)


def test_sessions_have_unique_ids_and_independent_question_specs() -> None:
    _, first = start_interview()
    _, second = start_interview()

    first.questions[0].spec.text = "Изменённый вопрос"

    assert first.interview_id != second.interview_id
    assert second.questions[0].spec.text == "Сколько вам лет?"


def test_empty_questionnaire_is_rejected() -> None:
    try:
        start_interview([])
    except ValueError as exc:
        assert "хотя бы один" in str(exc)
    else:
        raise AssertionError("Пустая анкета должна быть отклонена")


def test_finished_session_gets_completion_timestamp() -> None:
    questions = [QuestionSpec(id=1, kind="open", text="Расскажите", max_followups=0)]
    _, session = start_interview(questions)

    _, session = advance(session, "Согласен", InterviewLLM())
    reply, session = advance(session, "Содержательный ответ", InterviewLLM())

    assert session.finished
    assert session.completed_at is not None
    assert session.questions[0].answer == "Содержательный ответ"
    assert "завершён" in reply


def test_closed_questions_are_normalized_by_schema() -> None:
    questions = [
        QuestionSpec(id=1, kind="age", text="Возраст?"),
        QuestionSpec(id=2, kind="city", text="Город?"),
    ]
    _, session = start_interview(questions)

    _, session = advance(session, "Согласен", InterviewLLM())
    _, session = advance(session, "Мне тридцать", InterviewLLM())
    _, session = advance(session, "Живу в Москве", InterviewLLM())

    assert session.collected_answers() == {1: "30", 2: "Москва"}
    assert session.finished


def test_declined_consent_collects_no_answers() -> None:
    _, session = start_interview()

    reply, session = advance(session, "Не согласен", InterviewLLM())

    assert session.declined
    assert not session.finished
    assert not session.collected_answers()
    assert "не собираются" in reply


def test_initial_answer_and_single_followup_are_stored_separately() -> None:
    questions = [
        QuestionSpec(id=1, kind="open", text="Что не нравится?", max_followups=2)
    ]
    _, session = start_interview(questions)

    _, session = advance(session, "Согласен", OneFollowupLLM())
    followup, session = advance(session, "Высокая комиссия", OneFollowupLLM())

    state = session.questions[0]
    assert followup == "Можете привести пример?"
    assert state.initial_answer == "Высокая комиссия"
    assert state.answer == "Высокая комиссия"
    assert state.followups[0].reason == "missing_context"
    assert state.followups[0].answer is None

    _, session = advance(session, "При оплате ЖКХ", OneFollowupLLM())

    assert session.finished
    assert state.answer == "Высокая комиссия При оплате ЖКХ"
    assert state.followups_asked == 1
    assert state.followups[0].answer == "При оплате ЖКХ"
    received = [event for event in session.events if event.kind == "answer_received"]
    assert [event.details["response_kind"] for event in received] == [
        "initial",
        "followup",
    ]
