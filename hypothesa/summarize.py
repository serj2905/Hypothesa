"""Суммаризация открытых ответов через structured output + Judge 2 (Faithfulness).

Поток:
  1. Генератор (Qwen) сжимает ответ в OpenAnswer{items} по жёсткой схеме.
  2. Judge 2 (модель ДРУГОГО семейства) проверяет, что каждый item подтверждается
     исходным ответом. При галлюцинации — один повтор с явным указанием на проблему.

Это закрывает баг 2 (галлюцинации суммаризатора) из CLAUDE.md.
"""

from __future__ import annotations

from . import config
from .llm import LLMClient
from .schemas import FaithfulnessVerdict, OpenAnswer

_SUMMARIZE_SYSTEM = (
    "Ты выделяешь из ответа респondента отдельные аспекты дословно близко к тексту. "
    "СТРОГО запрещено добавлять то, чего нет в ответе. Если конкретных аспектов нет "
    "(«всё нравится», «ничего», «нет») — верни пустой список. "
    "Не интерпретируй, не обобщай, не придумывай."
)

_JUDGE_SYSTEM = (
    "Ты строгий проверяющий (faithfulness judge). На вход — исходный ответ респондента "
    "и список извлечённых из него аспектов. Верни faithful=true, только если КАЖДЫЙ "
    "аспект напрямую подтверждается исходным ответом. Любой аспект, которого нет в "
    "ответе, перечисли в unsupported_claims и верни faithful=false."
)


def summarize_open_answer(
    raw_answer: str,
    generator: LLMClient | None = None,
    judge: LLMClient | None = None,
    max_retries: int = 1,
) -> tuple[OpenAnswer, FaithfulnessVerdict]:
    """Суммаризировать открытый ответ и проверить на галлюцинации.

    Возвращает финальную суммаризацию и вердикт судьи. Если после ретраев
    faithfulness не достигнут — отдаёт последнюю попытку с faithful=false,
    чтобы вызывающий код мог отбросить/пометить запись.
    """
    generator = generator or LLMClient(model=config.LLM_MODEL)
    judge = judge or LLMClient(model=config.JUDGE_MODEL)

    messages = [
        {"role": "system", "content": _SUMMARIZE_SYSTEM},
        {"role": "user", "content": raw_answer},
    ]

    verdict = FaithfulnessVerdict(faithful=False)
    summary = OpenAnswer()

    for attempt in range(max_retries + 1):
        summary = generator.structured(OpenAnswer, messages)
        verdict = judge_faithfulness(raw_answer, summary, judge)
        if verdict.faithful:
            break
        # Подсказываем генератору, что именно он выдумал, и просим повтор.
        messages.append(
            {
                "role": "user",
                "content": (
                    "В прошлом ответе ты добавил аспекты, которых нет в исходном тексте: "
                    f"{verdict.unsupported_claims}. Повтори, оставив только то, что явно "
                    "сказано респондентом."
                ),
            }
        )

    return summary, verdict


def judge_faithfulness(
    raw_answer: str,
    summary: OpenAnswer,
    judge: LLMClient | None = None,
) -> FaithfulnessVerdict:
    judge = judge or LLMClient(model=config.JUDGE_MODEL)
    user = (
        f"Исходный ответ респондента:\n{raw_answer}\n\n"
        f"Извлечённые аспекты:\n{summary.items}"
    )
    return judge.structured(
        FaithfulnessVerdict,
        [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
