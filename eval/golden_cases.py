"""Golden set — реальные кейсы из прототипа, на которых видно баги.

Источник: DHA_hackathon/data/.../results.csv. Это первые тест-кейсы eval-harness:
если новый structured-output-суммаризатор отрабатывает их чисто, значит баг
галлюцинаций и кривого формата закрыт.

Поле `banned_substrings` — слова, которых НЕ должно быть в корректной суммаризации
(в прототипе GigaChat подставлял сюда «Python», «библиотеки» и т.п.).
`expect_empty` — ответ-заглушка, корректная суммаризация = пустой список.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GoldenCase:
    name: str
    raw_answer: str
    banned_substrings: list[str] = field(default_factory=list)
    expect_empty: bool = False


# Кейсы галлюцинаций — GigaChat выдумывал про Python вместо банка.
HALLUCINATION_CASES: list[GoldenCase] = [
    GoldenCase(
        name="python_hallucination_negatives",
        raw_answer="Клиентоориентированность Перерасчет бонусов спасибо Количество офисов Нет Нет",
        banned_substrings=["python", "библиотек", "синтаксис", "код"],
    ),
    GoldenCase(
        name="python_hallucination_positives",
        raw_answer="Скорость, удобство Нет 👍",
        banned_substrings=["python"],
    ),
]

# Кейсы-заглушки — корректная суммаризация должна вернуть пустой список.
EMPTY_CASES: list[GoldenCase] = [
    GoldenCase(name="empty_nichego", raw_answer="Ничего", expect_empty=True),
    GoldenCase(name="empty_vse_nravitsya", raw_answer="Все нравится Ага", expect_empty=True),
]

# Нормальные содержательные ответы — суммаризация должна выделить аспекты.
CONTENT_CASES: list[GoldenCase] = [
    GoldenCase(
        name="zhkh_commission",
        raw_answer="Комиссия при оплате ЖКХ Целиком вся коммисия, сталкиваюсь каждый месяц Это все",
        banned_substrings=["python"],
    ),
    GoldenCase(
        name="app_and_queues",
        raw_answer=(
            "Отсутствие нормального мобильного приложения Отсутствие приложения "
            "Длинные очереди в отделениях Качество, надежность услуг Нет"
        ),
        banned_substrings=["python"],
    ),
]

ALL_CASES: list[GoldenCase] = (
    HALLUCINATION_CASES + EMPTY_CASES + CONTENT_CASES
)
