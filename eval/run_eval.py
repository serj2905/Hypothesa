"""Eval-harness — измеритель качества суммаризации.

Гоняет golden-кейсы через новый structured-output-суммаризатор и считает:
  - format_valid_rate  — доля валидных JSON по схеме (цель: 100%)
  - hallucination_rate — доля кейсов с banned-словами или faithful=false
  - empty_correct_rate — корректно ли заглушки дали пустой список

Запуск на 5080-боксе (нужен поднятый Ollama):
    python -m eval.run_eval
"""

from __future__ import annotations

from eval.golden_cases import ALL_CASES, GoldenCase
from hypothesa.summarize import summarize_open_answer


def _check_case(case: GoldenCase) -> dict:
    result = {"name": case.name, "format_valid": False, "faithful": None, "passed": False}
    try:
        summary, verdict = summarize_open_answer(case.raw_answer)
    except Exception as exc:  # noqa: BLE001 — на этапе harness ловим всё, чтобы не падать на одном кейсе
        result["error"] = repr(exc)
        return result

    result["format_valid"] = True  # дошли сюда => схема провалидирована
    result["faithful"] = verdict.faithful
    result["items"] = summary.items

    text = " ".join(summary.items).lower()
    has_banned = any(b.lower() in text for b in case.banned_substrings)

    if case.expect_empty:
        result["passed"] = len(summary.items) == 0
    else:
        result["passed"] = (not has_banned) and verdict.faithful

    result["has_banned"] = has_banned
    return result


def main() -> None:
    results = [_check_case(c) for c in ALL_CASES]
    n = len(results)

    format_valid = sum(r["format_valid"] for r in results)
    passed = sum(r["passed"] for r in results)
    hallucinated = sum(
        1 for r in results if r.get("has_banned") or r.get("faithful") is False
    )

    print(f"\n{'кейс':<32} {'format':<8} {'faithful':<10} {'pass':<6}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['name']:<32} "
            f"{'ok' if r['format_valid'] else 'FAIL':<8} "
            f"{str(r.get('faithful')):<10} "
            f"{'✓' if r['passed'] else '✗':<6}"
        )
        if r.get("error"):
            print(f"    error: {r['error']}")
        elif not r["passed"]:
            print(f"    items: {r.get('items')}")

    print("-" * 60)
    print(f"format_valid_rate:  {format_valid}/{n} = {format_valid / n:.0%}")
    print(f"passed:             {passed}/{n} = {passed / n:.0%}")
    print(f"hallucination_rate: {hallucinated}/{n} = {hallucinated / n:.0%}")


if __name__ == "__main__":
    main()
