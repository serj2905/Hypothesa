"""Запуск golden-eval с защитой от пустых ответов и регрессий.

Обычный запуск всегда пишет отдельный отчёт в ``eval/results``. Эталонный
``eval/baseline.json`` меняется только с явным флагом ``--update-baseline``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from eval.golden_cases import ALL_CASES, GoldenCase
from eval.scoring import CaseScore, aggregate_scores, failed_case, score_case
from hypothesa import config
from hypothesa.llm import LLMClient, release_llm_client
from hypothesa.schemas import FaithfulnessVerdict, OpenAnswer
from hypothesa.summarize import (
    generate_open_answer,
    judge_faithfulness,
)

EVAL_DIR = Path(__file__).parent
BASELINE_PATH = EVAL_DIR / "baseline.json"
RESULTS_DIR = EVAL_DIR / "results"
ClientFactory = Callable[[], LLMClient]

QUALITY_GATES = {
    "format_valid_rate": (">=", 1.0),
    "passed_rate": (">=", 0.70),
    "empty_correct_rate": (">=", 0.90),
    "content_non_empty_rate": (">=", 0.90),
    "mean_concept_recall": (">=", 0.60),
    "mean_concept_precision": (">=", 0.60),
    "hallucination_rate": ("<=", 0.30),
}

REGRESSION_TOLERANCE = {
    "passed_rate": 0.05,
    "mean_concept_recall": 0.05,
    "mean_concept_precision": 0.05,
    "hallucination_rate": 0.05,
}


@dataclass
class _PendingCase:
    case: GoldenCase
    summary: OpenAnswer | None = None
    verdict: FaithfulnessVerdict | None = None
    unsupported_claims: list[str] | None = None
    error: Exception | None = None


def _generate_phase(items: Sequence[_PendingCase], factory: ClientFactory) -> None:
    generator = factory()
    try:
        for item in items:
            try:
                item.summary = generate_open_answer(
                    item.case.raw_answer,
                    generator,
                    unsupported_claims=item.unsupported_claims,
                )
            except Exception as exc:  # один кейс не останавливает eval
                item.error = exc
    finally:
        release_llm_client(generator, "generator", close=True)


def _judge_phase(items: Sequence[_PendingCase], factory: ClientFactory) -> None:
    judge = factory()
    try:
        for item in items:
            if item.error is not None or item.summary is None:
                continue
            try:
                item.verdict = judge_faithfulness(
                    item.case.raw_answer,
                    item.summary,
                    judge,
                )
            except Exception as exc:
                item.error = exc
    finally:
        release_llm_client(judge, "judge", close=True)


def evaluate_cases(
    cases: Sequence[GoldenCase],
    *,
    generator_factory: ClientFactory | None = None,
    judge_factory: ClientFactory | None = None,
) -> list[CaseScore]:
    """Оценить корпус пакетно, не чередуя модели в VRAM для каждого кейса."""
    generator_factory = generator_factory or (lambda: LLMClient(model=config.LLM_MODEL))
    judge_factory = judge_factory or (lambda: LLMClient(model=config.JUDGE_MODEL))
    pending = [_PendingCase(case=case) for case in cases]

    _generate_phase(pending, generator_factory)
    _judge_phase(pending, judge_factory)

    retry = [
        item
        for item in pending
        if item.error is None and item.verdict is not None and not item.verdict.faithful
    ]
    if retry:
        for item in retry:
            item.unsupported_claims = item.verdict.unsupported_claims
            item.summary = None
            item.verdict = None
        _generate_phase(retry, generator_factory)
        _judge_phase(retry, judge_factory)

    scores = []
    for item in pending:
        if item.error is not None:
            scores.append(failed_case(item.case, item.error))
        elif item.summary is None or item.verdict is None:
            scores.append(
                failed_case(
                    item.case,
                    RuntimeError("Eval-кейс завершился без summary или verdict."),
                )
            )
        else:
            scores.append(score_case(item.case, item.summary, item.verdict))
    return scores


def evaluate_case(case: GoldenCase) -> CaseScore:
    return evaluate_cases([case])[0]


def build_report(scores: list[CaseScore]) -> dict:
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "generator_model": config.LLM_MODEL,
        "judge_model": config.JUDGE_MODEL,
        "n_cases": len(scores),
        "metrics": aggregate_scores(scores),
        "cases": [score.as_dict() for score in scores],
    }


def check_quality_gates(metrics: dict[str, float]) -> list[str]:
    failures = []
    for name, (operator, target) in QUALITY_GATES.items():
        value = metrics[name]
        passed = value >= target if operator == ">=" else value <= target
        if not passed:
            failures.append(f"{name}: {value:.1%} (нужно {operator} {target:.1%})")
    return failures


def check_regressions(report: dict, baseline: dict) -> list[str]:
    current = report["metrics"]
    previous = baseline["metrics"]
    failures = []
    for name, tolerance in REGRESSION_TOLERANCE.items():
        delta = current[name] - previous[name]
        regressed = delta > tolerance if name == "hallucination_rate" else delta < -tolerance
        if regressed:
            failures.append(f"{name}: {previous[name]:.1%} -> {current[name]:.1%}")
    return failures


def write_report(report: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = RESULTS_DIR / f"eval-{stamp}.json"
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    output_path.write_text(serialized, encoding="utf-8")
    return output_path


def write_baseline(report: dict) -> None:
    BASELINE_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def print_report(scores: list[CaseScore], metrics: dict[str, float]) -> None:
    print(f"\n{'кейс':<38} {'format':<7} {'faithful':<9} {'P':<7} {'R':<7} pass")
    print("-" * 86)
    for score in scores:
        recall = "—" if score.concept_recall is None else f"{score.concept_recall:.0%}"
        precision = "—" if score.concept_precision is None else f"{score.concept_precision:.0%}"
        print(
            f"{score.name:<38} "
            f"{'ok' if score.format_valid else 'FAIL':<7} "
            f"{str(score.faithful):<9} {precision:<7} {recall:<7} "
            f"{'PASS' if score.passed else 'FAIL'}"
        )
        if score.error:
            print(f"    error: {score.error}")
        elif not score.passed:
            print(f"    items: {score.items}")
    print("-" * 86)
    for name, value in metrics.items():
        print(f"{name:<25} {value:.1%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Явно заменить baseline текущим результатом после ручной проверки.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scores = evaluate_cases(ALL_CASES)
    report = build_report(scores)
    output_path = write_report(report)
    print_report(scores, report["metrics"])
    print(f"\nРезультат: {output_path}")

    failures = check_quality_gates(report["metrics"])
    if BASELINE_PATH.exists() and not args.update_baseline:
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        failures.extend(check_regressions(report, baseline))
    elif not args.update_baseline:
        print("Baseline отсутствует; создайте его только после ручной проверки запуска.")

    if failures:
        print("\nEval не прошёл:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    if args.update_baseline:
        write_baseline(report)
        print(f"\nBaseline обновлён: {BASELINE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
