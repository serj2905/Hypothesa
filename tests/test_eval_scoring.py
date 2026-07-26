from eval.golden_cases import ALL_CASES, GoldenCase, concepts
from eval.run_eval import evaluate_cases
from eval.scoring import aggregate_scores, concept_precision, concept_recall, score_case
from hypothesa.schemas import FaithfulnessVerdict, OpenAnswer

FAITHFUL = FaithfulnessVerdict(faithful=True)


def test_always_empty_summarizer_cannot_pass_content_cases() -> None:
    scores = [score_case(case, OpenAnswer(items=[]), FAITHFUL) for case in ALL_CASES]

    assert sum(score.passed for score in scores) == sum(
        case.expect_empty for case in ALL_CASES
    )
    assert aggregate_scores(scores)["content_non_empty_rate"] == 0.0


def test_concept_recall_accepts_normalization_and_alternatives() -> None:
    case = GoldenCase(
        name="cashback",
        raw_answer="Кэшбэк рублями",
        expected_concepts=concepts("кэшбэк|кешбэк", "рубл"),
    )

    assert concept_recall(case, ["Кешбэк начисляется в рублях."]) == 1.0
    assert concept_precision(case, ["Кешбэк начисляется в рублях."]) == 1.0


def test_content_case_requires_minimum_concept_coverage() -> None:
    case = GoldenCase(
        name="two-concepts",
        raw_answer="Комиссия при оплате ЖКХ",
        expected_concepts=concepts("комисс", "жкх"),
        min_concept_recall=1.0,
    )

    incomplete = score_case(case, OpenAnswer(items=["Комиссия"]), FAITHFUL)
    complete = score_case(
        case,
        OpenAnswer(items=["Комиссия при оплате ЖКХ"]),
        FAITHFUL,
    )

    assert not incomplete.passed
    assert complete.passed


def test_banned_hallucination_fails_even_with_full_coverage() -> None:
    case = GoldenCase(
        name="hallucination",
        raw_answer="Удобное приложение",
        banned_substrings=["python"],
        expected_concepts=concepts("приложен"),
    )

    score = score_case(
        case,
        OpenAnswer(items=["Удобное приложение на Python"]),
        FAITHFUL,
    )

    assert score.has_banned
    assert not score.passed


class _Generator:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def structured(self, schema, messages, temperature=0.0):
        self.calls.append("generate")
        answer = messages[1]["content"]
        return OpenAnswer(items=[answer.removeprefix("<answer>").removesuffix("</answer>")])

    def unload(self) -> None:
        self.calls.append("unload_generator")

    def close(self) -> None:
        self.calls.append("close_generator")


class _Judge:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def structured(self, schema, messages, temperature=0.0):
        self.calls.append("judge")
        return FAITHFUL

    def unload(self) -> None:
        self.calls.append("unload_judge")

    def close(self) -> None:
        self.calls.append("close_judge")


def test_eval_separates_generator_and_judge_phases() -> None:
    calls: list[str] = []
    cases = [
        GoldenCase(
            name="first",
            raw_answer="Первый",
            expected_concepts=concepts("перв"),
        ),
        GoldenCase(
            name="second",
            raw_answer="Второй",
            expected_concepts=concepts("втор"),
        ),
    ]

    scores = evaluate_cases(
        cases,
        generator_factory=lambda: _Generator(calls),
        judge_factory=lambda: _Judge(calls),
    )

    assert calls == [
        "generate",
        "generate",
        "unload_generator",
        "close_generator",
        "judge",
        "judge",
        "unload_judge",
        "close_judge",
    ]
    assert all(score.passed for score in scores)
