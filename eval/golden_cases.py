"""Golden set с репрезентативными кейсами пилота.

Ответы сохранены как обычные структуры данных, поэтому регрессионная оценка не
зависит от внешних файлов и сериализованных классов. Если structured-output
суммаризатор отрабатывает эти кейсы чисто, защита от галлюцинаций и нарушения
формата продолжает работать.

Поле `banned_substrings` — слова, которых НЕ должно быть в корректной суммаризации
(в прототипе GigaChat подставлял сюда «Python», «библиотеки», «переводчик» и т.п.).
`expect_empty` — ответ-заглушка без конкретики, корректная суммаризация = пустой список.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def concepts(*groups: str) -> list[tuple[str, ...]]:
    """Собрать эталонные понятия; ``|`` разделяет допустимые формулировки."""
    return [tuple(part.strip() for part in group.split("|")) for group in groups]


@dataclass
class GoldenCase:
    name: str
    raw_answer: str
    banned_substrings: list[str] = field(default_factory=list)
    expect_empty: bool = False
    expected_concepts: list[tuple[str, ...]] = field(default_factory=list)
    min_concept_recall: float = 0.6
    min_concept_precision: float = 0.5


# Кейсы галлюцинаций — GigaChat выдумывал про Python/переводчиков вместо банка.
HALLUCINATION_CASES: list[GoldenCase] = [
    GoldenCase(
        name="python_hallucination_negatives",
        raw_answer="Клиентоориентированность Перерасчет бонусов спасибо Количество офисов Нет Нет",
        banned_substrings=["python", "библиотек", "синтаксис", "код"],
        expected_concepts=concepts("клиентоориент", "перерасчет|бонус", "офис"),
    ),
    GoldenCase(
        name="python_hallucination_positives",
        raw_answer="Скорость, удобство Нет 👍",
        banned_substrings=["python"],
        expected_concepts=concepts("скорост", "удобств"),
    ),
    GoldenCase(
        name="stickers_vk_python_hallucination",
        raw_answer=(
            "Стикеры в вк Нет, я имел ввиду то, что раз в несколько месяцев мне "
            "дарят набор стикеров от сбербанка в социальной сети \"Вконтакте\" Это все"
        ),
        banned_substrings=["python"],
        expected_concepts=concepts("стикер", "вконтакт|вк"),
    ),
    GoldenCase(
        name="phone_transfer_translator_hallucination",
        raw_answer="Перевод по телефону Комиссия за перевод большая",
        banned_substrings=["переводчик", "на разных языках", "медицинских"],
        expected_concepts=concepts("перевод по телефон", "комисс"),
    ),
]

# Кейсы-заглушки — корректная суммаризация должна вернуть пустой список.
EMPTY_CASES: list[GoldenCase] = [
    GoldenCase(name="empty_nichego", raw_answer="Ничего", expect_empty=True),
    GoldenCase(name="empty_vse_nravitsya", raw_answer="Все нравится Ага", expect_empty=True),
    GoldenCase(name="empty_vse", raw_answer="Все", expect_empty=True),
    GoldenCase(name="empty_vse_ustraivaet", raw_answer="Все устраивает Да", expect_empty=True),
    GoldenCase(
        name="empty_net_narekaniy",
        raw_answer="Нет нареканий по оказываемым услугам",
        expect_empty=True,
    ),
    GoldenCase(
        name="empty_ne_znau_ne_polzovalas",
        raw_answer="Я не знаю \nЯ не пользовалась Так карты легли Нет",
        expect_empty=True,
    ),
]

# Нормальные содержательные ответы — суммаризация должна выделить аспекты.
CONTENT_CASES: list[GoldenCase] = [
    GoldenCase(
        name="zhkh_commission",
        raw_answer="Комиссия при оплате ЖКХ Целиком вся коммисия, сталкиваюсь каждый месяц Это все",
        banned_substrings=["python"],
        expected_concepts=concepts("комисс|коммис", "жкх"),
    ),
    GoldenCase(
        name="app_and_queues",
        raw_answer=(
            "Отсутствие нормального мобильного приложения Отсутствие приложения "
            "Длинные очереди в отделениях Качество, надежность услуг Нет"
        ),
        banned_substrings=["python"],
        expected_concepts=concepts("приложен", "очеред", "качеств|надежност"),
    ),
    GoldenCase(
        name="transfer_fee_limit",
        raw_answer=(
            "Снятие денег за переводы внутри Сбербанка и людям Сбербанка Лимит "
            "стоит от 50 то Ничего Ок"
        ),
        banned_substrings=["python"],
        expected_concepts=concepts("снятие денег|комисс", "перевод", "лимит|50"),
    ),
    GoldenCase(
        name="missing_self_transfer_button",
        raw_answer=(
            "Нет кнопки «перевести себе» в другой банк Часто Нет, нажимаю больше "
            "кнопок в вашем Спасибо"
        ),
        banned_substrings=["python"],
        expected_concepts=concepts("нет кнопк|отсутств", "перевести себе", "другой банк"),
    ),
    GoldenCase(
        name="spam_sms",
        raw_answer="Спам смс Надоедают Об операциях Нет Да",
        banned_substrings=["python"],
        expected_concepts=concepts("спам", "смс"),
    ),
    GoldenCase(
        name="spam_calls",
        raw_answer="Спам-звонки Еженедельно Хз Далее",
        banned_substrings=["python"],
        expected_concepts=concepts("спам", "звон"),
    ),
    GoldenCase(
        name="gov_payment_commission_competitor",
        raw_answer=(
            "Комиссия за оплату гос учреждений, при том что в Данил банках её нет "
            "(ЖКХ, садик и тд) Тинькофф Остальное вполне устраивает Ок"
        ),
        banned_substrings=["python"],
        expected_concepts=concepts("комисс", "гос учрежден|жкх|садик", "тинькофф"),
    ),
    GoldenCase(
        name="office_hours_closure",
        raw_answer=(
            "Закрытие офисов в 18.00 Приходится отпрашиваться с работы, что бы "
            "успеть в банк. Бывают вопросы, которые невозможно решить через банк "
            "он-лайн. И в выходные работает только одни банк. Многие можно сделать "
            "через сбер он-лайн. Персонал  работающий в офисах банка."
        ),
        banned_substrings=["python"],
        expected_concepts=concepts("18.00|18:00", "отпраш", "выходн", "он-лайн|онлайн"),
    ),
    GoldenCase(
        name="mortgage_rate_and_numeric_limit",
        raw_answer=(
            "Высокие ипотечные ставки Как у работников IT компаний Ограничение "
            "переводов без комиссий Переводы без комиссий до 50000 тысяч\n"
            "Обычно не могу отследить когда уже превысил лимит Я не пользуюсь "
            "другими услугами, только карта мир"
        ),
        banned_substrings=["python"],
        expected_concepts=concepts("ипотечн", "ставк", "лимит|ограничен", "50000|50 000"),
    ),
    GoldenCase(
        name="deposits_cashback_long_emoji",
        raw_answer=(
            "Мало вариантов вкладов, кэшбэк Сберспасибо(а хотелось бы рублями) "
            "слишком заморочен и от этого бесполезен. Вклад с пополнением и "
            "снятием с несколько большим процентом, потому что менее 2 % не "
            "интересно, а если чуть больше процент, то сразу стартовая сумма "
            "больше и снятие отменено… в общем поэтому и не привлекают именно в "
            "Сбере вклады.\nКэшбэк слишком ограничивает траты- именно в приложении "
            "и только по некоторым категориям… и лень возиться, проще переложить "
            "деньги на карту другого банка, где кэшбэк в рублях🤗 Пару акций лежат "
            "на сбере, но тоже, чисто символически, не очень удобно прыгать по "
            "разным банкам, а тут немного разочаровалась, поэтому практически не "
            "пользуюсь теперь 👌"
        ),
        banned_substrings=["python"],
        expected_concepts=concepts(
            "вклад", "кэшбэк|кешбэк", "сберспасибо|спасибо", "рубл", "процент"
        ),
    ),
    GoldenCase(
        name="app_convenience_long_emoji",
        raw_answer=(
            "Нравится удобство приложения, интерфэйс Нравится запоминание "
            "последних контактов кому отправляла средства, даже если это другой "
            "банк, нравится, что всё, что может понадобится оперативно - камера "
            "куаркода например, находится на главной, нравится что легко "
            "отследить траты не ведя раскопки в приложении, и плюс анализ "
            "финансов 👍 Получше продумать раздел кэшбэка 😌 и сразу в разы чаще "
            "пользоваться сбером начну. 👌"
        ),
        banned_substrings=["python"],
        expected_concepts=concepts(
            "удобств|интерф", "контакт", "куаркод|qr", "трат", "кэшбэк|кешбэк"
        ),
    ),
    GoldenCase(
        name="fast_no_ads_many_atms",
        raw_answer=(
            "Быстро, без лишней рекламы Много банкоматов Нет Чтобы все они имели "
            "возможность снимать и пополнять и в них были деньги и размен Ок"
        ),
        banned_substrings=["python"],
        expected_concepts=concepts("быстр", "реклам", "банкомат", "снимат|пополнят"),
    ),
    GoldenCase(
        name="convenient_app_view_cards",
        raw_answer="Удобный интерфейс приложения Удобно смотреть все свои карты и совершать переводы Нет",
        banned_substrings=["python"],
        expected_concepts=concepts("интерфейс|приложен", "карт", "перевод"),
    ),
    GoldenCase(
        name="self_employed_tax_glitch_occasional",
        raw_answer=(
            "Иногда подвисает система для самозанятых, невозможно оплатить налог "
            "из приложения Сбера. Не часто, в основном в конце дня. Больше "
            "проблем нет."
        ),
        banned_substrings=["python"],
        expected_concepts=concepts("подвиса", "самозанят", "налог", "конц дня|конце дня"),
    ),
    GoldenCase(
        name="salary_perks_deposits_investment",
        raw_answer=(
            "Доступность, наличие \"спасибо\", особые предложения для зарплатных "
            "клиентов. Вклады с хорошим процентом, наличие площадки для "
            "инвестиций. Нет, все хорошо. Спасибо."
        ),
        banned_substrings=["python"],
        expected_concepts=concepts("спасибо", "зарплат", "вклад", "процент", "инвестиц"),
    ),
]

ALL_CASES: list[GoldenCase] = (
    HALLUCINATION_CASES + EMPTY_CASES + CONTENT_CASES
)
