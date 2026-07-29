"""Идемпотентно наполнить отдельный survey человекоподобными synthetic-интервью.

Seed не вызывает LLM: raw-ответы и summary строятся вместе, а каждый summary-item
проверяется на присутствие в исходном ответе. Synthetic-корпус по умолчанию
изолирован суффиксом ``-synthetic-v1`` и отрицательными user_id.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from hypothesa import config
from hypothesa.interview import FollowupTurn, InterviewSession, start_interview
from hypothesa.schemas import OpenAnswer
from hypothesa.storage import STATUS_SUMMARIZED, Storage
from hypothesa.summarize import CompletedInterview, OpenAnswerResult

NEGATIVE_ITEMS: dict[str, tuple[str, ...]] = {
    "mobile_app": (
        "Приложение иногда зависает при входе.",
        "После обновления мобильный банк стал заметно медленнее.",
        "Код подтверждения иногда приходит только со второй попытки.",
        "История операций в приложении загружается слишком долго.",
        "Оплата по QR-коду пару раз завершалась ошибкой.",
        "Не всегда получается войти в приложение по отпечатку.",
        "Пуши об операциях приходят с задержкой.",
        "Раздел с кэшбэком в приложении сложно найти.",
    ),
    "fees": (
        "Комиссия за перевод отображается только перед подтверждением.",
        "Не нравится плата за уведомления по карте.",
        "Тарифы на обслуживание карты объяснены слишком сложно.",
        "За перевод в другой банк списали неожиданную комиссию.",
        "Стоимость годового обслуживания оказалась выше, чем я ожидал.",
        "Не сразу понятно, какие услуги входят в подписку.",
        "Комиссия за оплату ЖКХ кажется завышенной.",
        "Лимит бесплатных переводов трудно отслеживать.",
    ),
    "support": (
        "В чате трудно добраться от бота до живого оператора.",
        "Ответ поддержки пришлось ждать почти двадцать минут.",
        "Операторы несколько раз переводили обращение друг другу.",
        "Поддержка закрыла обращение до решения вопроса.",
        "По телефону дали два разных ответа на один вопрос.",
        "На горячей линии долго играет музыка ожидания.",
        "В чате приходится заново пересказывать проблему.",
        "Поддержка не сразу разобралась с блокировкой карты.",
    ),
    "branches": (
        "В отделении вечером обычно большая очередь.",
        "В офисе работало только два окна из пяти.",
        "Сотрудник настойчиво предлагал дополнительную страховку.",
        "Ближайшее отделение закрывается слишком рано.",
        "Получение новой карты заняло больше сорока минут.",
        "В выходные трудно найти работающее отделение.",
        "Электронная очередь в офисе иногда сбивается.",
        "В отделении не смогли сразу объяснить условия кредита.",
    ),
    "atm": (
        "Банкомат рядом с домом часто не принимает наличные.",
        "Вечером в ближайшем банкомате закончились деньги.",
        "Банкомат долго пересчитывает внесённые купюры.",
        "Один раз банкомат завис во время пополнения.",
        "Не все банкоматы позволяют снять мелкую сумму.",
        "Приходится искать банкомат с функцией внесения наличных.",
        "Экран старого банкомата реагирует с задержкой.",
        "После ошибки банкомата возврат денег занял несколько дней.",
    ),
    "cashback": (
        "Категории кэшбэка меняются слишком часто.",
        "Бонусы Спасибо нельзя потратить в привычных магазинах.",
        "Хотелось бы получать кэшбэк рублями, а не бонусами.",
        "Условия начисления бонусов написаны непонятно.",
        "Повышенный кэшбэк действует на слишком узкие категории.",
        "В приложении неудобно проверять срок действия бонусов.",
        "За последнюю покупку бонусы начислились не сразу.",
        "Не хватает понятного расчёта кэшбэка по каждой покупке.",
    ),
    "transfers": (
        "Перевод между своими счетами спрятан глубоко в меню.",
        "Неудобно отслеживать остаток лимита бесплатных переводов.",
        "Перевод по номеру телефона иногда уходит не на ту карту.",
        "Шаблоны переводов нельзя нормально переименовать.",
        "Подтверждение перевода иногда занимает слишком много времени.",
        "В истории сложно найти перевод конкретному человеку.",
        "Для перевода себе в другой банк нужно сделать много нажатий.",
        "Не всегда понятно, почему перевод оказался отклонён.",
    ),
}

POSITIVE_ITEMS: dict[str, tuple[str, ...]] = {
    "mobile_app": (
        "Приложение работает быстро и редко зависает.",
        "В мобильном банке понятный главный экран.",
        "Удобно видеть все карты и счета в одном месте.",
        "Нравится быстрый вход по отпечатку.",
        "Историю расходов легко фильтровать по категориям.",
        "Оплата по QR-коду находится прямо на главном экране.",
        "Приложение запоминает недавних получателей переводов.",
        "Уведомления об операциях обычно приходят сразу.",
    ),
    "transfers": (
        "Переводы по номеру телефона проходят почти мгновенно.",
        "Удобно переводить деньги между своими счетами.",
        "Частые переводы можно сохранить как шаблон.",
        "Перед подтверждением хорошо видна итоговая сумма.",
        "Большинство моих переводов проходит без комиссии.",
        "В истории легко найти нужный перевод.",
        "Статус перевода обновляется без задержки.",
        "Получатель сразу видит отправленные деньги.",
    ),
    "support": (
        "Оператор в чате быстро решил мой вопрос.",
        "Поддержка подробно объяснила причину блокировки.",
        "Сотрудник горячей линии разговаривал спокойно и вежливо.",
        "В чате удалось связаться с человеком за пару минут.",
        "Обращение не закрывали, пока проблема не решилась.",
        "Оператор помог вернуть ошибочно списанную комиссию.",
        "Поддержка перезвонила в обещанное время.",
        "Ответ сотрудника был понятным и без лишних терминов.",
    ),
    "branches": (
        "В отделении меня приняли почти без очереди.",
        "Сотрудник офиса подробно рассказал про условия карты.",
        "Удобно, что отделение находится рядом с домом.",
        "В офисе помогли решить вопрос за одно посещение.",
        "Сотрудники отделения общались вежливо.",
        "Электронная очередь двигалась быстро.",
        "Новую карту выдали за несколько минут.",
        "В отделении чисто и есть место для ожидания.",
    ),
    "atm": (
        "Рядом с домом есть несколько банкоматов.",
        "Банкоматы обычно работают круглосуточно.",
        "Наличные вносятся быстро и без ошибок.",
        "Удобно, что банкомат выдаёт купюры разного номинала.",
        "В приложении легко найти ближайший банкомат.",
        "Банкомат сразу печатает понятный чек.",
        "Снять деньги можно почти в любом районе.",
        "Большинство банкоматов поддерживает бесконтактную карту.",
    ),
    "cashback": (
        "Бонусы Спасибо начисляются за обычные покупки.",
        "Нравится выбирать категории повышенного кэшбэка.",
        "В приложении видно, сколько бонусов начислено за покупку.",
        "Бонусами удобно оплачивать часть заказа.",
        "Иногда попадаются действительно полезные предложения.",
        "История начисления бонусов отображается достаточно подробно.",
        "Категории кэшбэка подходят моим обычным расходам.",
        "Приятно получать дополнительные бонусы у партнёров.",
    ),
    "reliability": (
        "Платежи обычно проходят с первого раза.",
        "За несколько лет карта ни разу неожиданно не блокировалась.",
        "Банк сразу сообщает о подозрительной операции.",
        "Зарплата всегда приходит без задержек.",
        "Автоплатежи работают стабильно.",
        "Операции подтверждаются быстро и понятно.",
        "Сервис доступен даже поздно вечером.",
        "Чеки и подтверждения операций всегда сохраняются.",
    ),
}

CITIES = (
    "Москва",
    "Санкт-Петербург",
    "Казань",
    "Екатеринбург",
    "Новосибирск",
    "Самара",
    "Пермь",
    "Уфа",
    "Тула",
    "Ярославль",
)

FOLLOWUP_QUESTIONS = (
    "Можете привести конкретный пример?",
    "Как часто вы с этим сталкиваетесь?",
    "Что именно в этой ситуации для вас важно?",
)


@dataclass(frozen=True)
class SeedInterview:
    user_id: int
    session: InterviewSession
    result: CompletedInterview


@dataclass(frozen=True)
class SeedResult:
    inserted: int
    skipped: int


class SeedStorage(Protocol):
    def load_session(self, user_id, interview_id): ...

    def start_session(self, user_id, session) -> None: ...

    def save_session(self, user_id, session) -> None: ...

    def finalize_summary(self, user_id, session, result) -> None: ...


def _human_initial_answer(
    items: list[str],
    rng: random.Random,
    *,
    positive: bool,
) -> str:
    if not items:
        return rng.choice(
            (
                "Да вроде ничего конкретного не могу назвать.",
                "В целом всё нормально, без особых замечаний.",
                "Наверное, ничего такого не вспомню.",
            )
        )
    lowered = [item[0].lower() + item[1:] for item in items]
    if len(lowered) == 1:
        templates = (
            "Если коротко, {first}",
            "Наверное, главное — {first}",
            "Из того, что сразу приходит в голову: {first}",
        )
        if positive:
            templates += ("Больше всего нравится, что {first}",)
        else:
            templates += ("Больше всего раздражает, что {first}",)
        return rng.choice(templates).format(first=lowered[0])
    return rng.choice(
        (
            "Во-первых, {first} Ещё {second}",
            "Сразу два момента: {first} И ещё {second}",
            "{first} При этом {second}",
        )
    ).format(first=lowered[0], second=lowered[1])


def _pick_items(
    pools: dict[str, tuple[str, ...]],
    rng: random.Random,
) -> tuple[str, list[str]]:
    primary = rng.choice(list(pools))
    items = [rng.choice(pools[primary])]
    if rng.random() < 0.42:
        secondary = rng.choice([name for name in pools if name != primary])
        items.append(rng.choice(pools[secondary]))
    return primary, items


def _answer_result(
    pools: dict[str, tuple[str, ...]],
    rng: random.Random,
    *,
    positive: bool,
) -> tuple[str, str, list[FollowupTurn], OpenAnswerResult]:
    primary, spontaneous_items = _pick_items(pools, rng)
    if rng.random() < 0.08:
        spontaneous_items = []
    initial = _human_initial_answer(spontaneous_items, rng, positive=positive)

    followups: list[FollowupTurn] = []
    enriched_items = list(spontaneous_items)
    followup_answers: list[str] = []
    if spontaneous_items and rng.random() < 0.45:
        candidates = [item for item in pools[primary] if item not in spontaneous_items]
        detail = rng.choice(candidates)
        answer = f"Например, {detail[0].lower() + detail[1:]}"
        followups.append(
            FollowupTurn(
                question=rng.choice(FOLLOWUP_QUESTIONS),
                answer=answer,
                reason="clarification",
            )
        )
        followup_answers.append(answer)
        enriched_items.append(detail)

    raw_answer = " ".join([initial, *followup_answers])
    result = OpenAnswerResult(
        raw_answer=raw_answer,
        initial_answer=initial,
        followup_answers=followup_answers,
        summary=OpenAnswer(items=enriched_items),
        spontaneous_summary=OpenAnswer(items=spontaneous_items),
        faithful=True,
        spontaneous_faithful=True,
    )
    return raw_answer, initial, followups, result


def _assert_grounded(result: OpenAnswerResult) -> None:
    raw = result.raw_answer.casefold().replace("ё", "е")
    initial = (result.initial_answer or result.raw_answer).casefold().replace("ё", "е")
    for item in result.summary.items:
        assert item.casefold().replace("ё", "е") in raw
    spontaneous = result.spontaneous_summary or result.summary
    for item in spontaneous.items:
        assert item.casefold().replace("ё", "е") in initial


def build_seed_interview(
    index: int,
    *,
    survey_id: str,
    seed: int,
    started_at: datetime,
) -> SeedInterview:
    rng = random.Random(f"{seed}:{index}")
    _, session = start_interview(
        survey_id=survey_id,
        variant="control",
    )
    session.interview_id = uuid5(
        NAMESPACE_URL,
        f"hypothesa:synthetic-faithful:{survey_id}:{seed}:{index}",
    )
    session.started_at = started_at
    session.consent_given = True

    age = rng.randint(19, 67)
    city = rng.choice(CITIES)
    session.questions[0].answer = str(age)
    session.questions[1].answer = city

    negative_raw, negative_initial, negative_followups, negative_result = _answer_result(
        NEGATIVE_ITEMS, rng, positive=False
    )
    positive_raw, positive_initial, positive_followups, positive_result = _answer_result(
        POSITIVE_ITEMS, rng, positive=True
    )
    for question, raw, initial, followups in (
        (
            session.questions[2],
            negative_raw,
            negative_initial,
            negative_followups,
        ),
        (
            session.questions[3],
            positive_raw,
            positive_initial,
            positive_followups,
        ),
    ):
        question.answer = raw
        question.initial_answer = initial
        question.followups = followups
        question.followups_asked = len(followups)

    session.current_index = len(session.questions)
    session.finished = True
    session.completed_at = started_at + timedelta(minutes=rng.randint(3, 9))
    session.record_event("consent_granted")
    session.record_event("session_completed")

    _assert_grounded(negative_result)
    _assert_grounded(positive_result)
    result = CompletedInterview(
        age=age,
        city=city,
        open_answers={
            3: negative_result,
            4: positive_result,
        },
    )
    user_id = -8_000_000_000_000_000 - seed * 100_000 - index
    return SeedInterview(user_id=user_id, session=session, result=result)


def build_corpus(
    count: int,
    *,
    survey_id: str,
    seed: int,
    now: datetime | None = None,
) -> list[SeedInterview]:
    now = now or datetime.now(UTC)
    return [
        build_seed_interview(
            index,
            survey_id=survey_id,
            seed=seed,
            started_at=now - timedelta(hours=count - index),
        )
        for index in range(count)
    ]


def seed_interviews(
    storage: SeedStorage,
    interviews: list[SeedInterview],
) -> SeedResult:
    inserted = 0
    skipped = 0
    for interview in interviews:
        existing = storage.load_session(
            interview.user_id,
            interview.session.interview_id,
        )
        if existing is not None:
            _, status = existing
            if status != STATUS_SUMMARIZED:
                raise RuntimeError(
                    f"Synthetic-интервью {interview.session.interview_id} "
                    f"уже существует со статусом {status!r}."
                )
            skipped += 1
            continue
        storage.start_session(interview.user_id, interview.session)
        storage.save_session(interview.user_id, interview.session)
        storage.finalize_summary(
            interview.user_id,
            interview.session,
            interview.result,
        )
        inserted += 1
    return SeedResult(inserted=inserted, skipped=skipped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--survey-id",
        default=f"{config.SURVEY_ID}-synthetic-v1",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Записать корпус в PostgreSQL; без флага показывается только план.",
    )
    parser.add_argument(
        "--allow-main-survey",
        action="store_true",
        help="Явно разрешить запись synthetic-данных в основной survey.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count должен быть положительным.")
    if args.survey_id == config.SURVEY_ID and not args.allow_main_survey:
        raise SystemExit(
            "Synthetic-данные нельзя писать в основной survey без --allow-main-survey."
        )

    interviews = build_corpus(
        args.count,
        survey_id=args.survey_id,
        seed=args.seed,
    )
    document_count = sum(
        len((answer.spontaneous_summary or answer.summary).items)
        for interview in interviews
        for answer in interview.result.open_answers.values()
    )
    print(
        f"Подготовлено synthetic-интервью: {len(interviews)}; "
        f"faithful BERTopic-документов: {document_count}; survey={args.survey_id}"
    )
    for interview in interviews[:3]:
        print(
            f"\nВозраст {interview.result.age}, город {interview.result.city}\n"
            f"  Не нравится: {interview.result.open_answers[3].raw_answer}\n"
            f"  Нравится: {interview.result.open_answers[4].raw_answer}"
        )

    if not args.apply:
        print("\nЗапись не выполнялась. Добавьте --apply для сохранения в PostgreSQL.")
        return 0

    storage = Storage()
    try:
        result = seed_interviews(storage, interviews)
        state = storage.topic_refresh_state(args.survey_id)
        actual_documents = len(storage.list_topic_documents(args.survey_id))
    finally:
        storage.close()
    print(
        f"\nЗаписано: {result.inserted}; уже существовало: {result.skipped}; "
        f"faithful-интервью в survey: {state.total_valid}; "
        f"BERTopic-документов: {actual_documents}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
