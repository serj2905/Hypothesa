from hypothesa.schemas import FaithfulnessVerdict, OpenAnswer
from hypothesa.summarize import generate_open_answer, judge_faithfulness


class Judge:
    def __init__(self, verdict: FaithfulnessVerdict) -> None:
        self.verdict = verdict
        self.calls = 0

    def structured(self, schema, messages, temperature=0.0):
        self.calls += 1
        return self.verdict


class Generator:
    def __init__(self, summaries: list[OpenAnswer]) -> None:
        self.summaries = iter(summaries)
        self.calls = 0
        self.messages = []

    def structured(self, schema, messages, temperature=0.0):
        self.calls += 1
        self.messages.append(messages)
        return next(self.summaries)


def test_empty_summary_is_faithful_without_judge_call() -> None:
    judge = Judge(
        FaithfulnessVerdict(
            faithful=False,
            unsupported_claims=["Фрагмент исходного ответа"],
        )
    )

    verdict = judge_faithfulness("Исходный ответ", OpenAnswer(items=[]), judge)

    assert verdict.faithful
    assert verdict.unsupported_claims == []
    assert judge.calls == 0


def test_judge_cannot_report_omitted_source_fact_as_unsupported() -> None:
    judge = Judge(
        FaithfulnessVerdict(
            faithful=False,
            unsupported_claims=["Об операциях"],
        )
    )

    verdict = judge_faithfulness(
        "Спам смс Об операциях Нет",
        OpenAnswer(items=["Спам смс"]),
        judge,
    )

    assert verdict.faithful
    assert verdict.unsupported_claims == []


def test_actual_unsupported_summary_item_is_preserved() -> None:
    judge = Judge(
        FaithfulnessVerdict(
            faithful=False,
            unsupported_claims=["Спам-звонки"],
        )
    )

    verdict = judge_faithfulness(
        "Спам смс",
        OpenAnswer(items=["Спам-звонки"]),
        judge,
    )

    assert not verdict.faithful
    assert verdict.unsupported_claims == ["Спам-звонки"]


def test_substantive_empty_draft_is_retried() -> None:
    generator = Generator(
        [
            OpenAnswer(items=[]),
            OpenAnswer(items=["Спам-звонки"]),
        ]
    )

    summary = generate_open_answer("Спам-звонки Хз Далее", generator)

    assert summary.items == ["Спам-звонки"]
    assert generator.calls == 2


def test_known_stub_does_not_trigger_content_retry() -> None:
    generator = Generator([OpenAnswer(items=[])])

    summary = generate_open_answer(
        "Я не знаю. Я не пользовалась. Так карты легли. Нет.",
        generator,
    )

    assert summary.items == []
    assert generator.calls == 1


def test_trailing_control_replies_are_removed_before_generation() -> None:
    generator = Generator([OpenAnswer(items=["Спам-звонки"])])

    generate_open_answer("Спам-звонки Еженедельно Хз Далее", generator)

    assert generator.messages[0][1]["content"] == "<answer>Спам-звонки Еженедельно</answer>"
