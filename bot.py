"""Telegram-бот поверх hypothesa/: тонкий async-слой, вся логика — в библиотеке.

Бот отвечает только за live-фазу диалога (interview.py) и знает только про
генератор (config.LLM_MODEL). Судью (config.JUDGE_MODEL) он не загружает
никогда — суммаризация завершённых интервью происходит отдельным batch-шагом,
см. `python -m hypothesa.worker`.

Storage и LLMClient синхронные, aiogram — асинхронный: каждый блокирующий
вызов оборачивается в asyncio.to_thread(), чтобы медленный ответ модели или
БД не замораживал бота для остальных пользователей.

Запуск:
    python bot.py
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from aiogram import Bot, Dispatcher
from aiogram.enums import ChatType
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from hypothesa import config
from hypothesa.experiments import assign_variant, pseudonymize_participant
from hypothesa.interview import advance, start_interview
from hypothesa.llm import LLMClient
from hypothesa.storage import ConcurrentSessionUpdate, Storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dp = Dispatcher()
storage = Storage()
generator = LLMClient(model=config.LLM_MODEL)
llm_semaphore = asyncio.Semaphore(config.BOT_MAX_CONCURRENT_LLM_REQUESTS)

_ERROR_REPLY = "Технические неполадки, попробуйте ответить ещё раз чуть позже."
_MAX_MESSAGE_LENGTH = 4000


@dataclass
class _LockEntry:
    lock: asyncio.Lock
    users: int = 0


class KeyedLockPool:
    """Сериализует сообщения участника и удаляет больше не нужные locks."""

    def __init__(self) -> None:
        self._entries: dict[int, _LockEntry] = {}

    @asynccontextmanager
    async def hold(self, key: int) -> AsyncIterator[None]:
        entry = self._entries.setdefault(key, _LockEntry(asyncio.Lock()))
        entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.users -= 1
            if entry.users == 0 and self._entries.get(key) is entry:
                self._entries.pop(key, None)

    def __len__(self) -> int:
        return len(self._entries)


chat_locks = KeyedLockPool()


def _participant_id(message: Message) -> int:
    if message.from_user is None:
        raise ValueError("У сообщения отсутствует отправитель.")
    return pseudonymize_participant(message.from_user.id, config.PARTICIPANT_SALT)


@dp.message(CommandStart())
async def on_start(message: Message) -> None:
    if message.chat.type != ChatType.PRIVATE:
        await message.answer("Опрос доступен только в личном чате с ботом.")
        return
    user_id = _participant_id(message)
    async with chat_locks.hold(user_id):
        latest_questionnaire = await asyncio.to_thread(
            storage.load_latest_questionnaire, config.SURVEY_ID
        )
        variant = (
            assign_variant(
                user_id,
                config.SURVEY_ID,
                adaptive_share=config.ADAPTIVE_SHARE,
            )
            if latest_questionnaire is not None
            else "control"
        )
        questionnaire = latest_questionnaire if variant == "adaptive" else None
        greeting, session = start_interview(
            questionnaire.questions if questionnaire else None,
            survey_id=config.SURVEY_ID,
            survey_version=questionnaire.version if questionnaire else 1,
            variant=variant,
        )
        await asyncio.to_thread(
            storage.start_session,
            user_id,
            session,
            replace_active=True,
        )
    await message.answer(greeting)


@dp.message(Command("delete_data"))
async def on_delete_data(message: Message) -> None:
    if message.chat.type != ChatType.PRIVATE:
        await message.answer("Удаление доступно только в личном чате с ботом.")
        return
    user_id = _participant_id(message)
    async with chat_locks.hold(user_id):
        deleted = await asyncio.to_thread(storage.delete_participant, user_id)
    if deleted:
        await message.answer("Ваши индивидуальные данные удалены.")
    else:
        await message.answer("Сохранённых индивидуальных данных не найдено.")


@dp.message()
async def on_message(message: Message) -> None:
    if message.text is None:
        await message.answer("Пожалуйста, отвечайте текстом.")
        return
    if message.chat.type != ChatType.PRIVATE:
        await message.answer("Опрос доступен только в личном чате с ботом.")
        return
    if len(message.text) > _MAX_MESSAGE_LENGTH:
        await message.answer(f"Ответ слишком длинный. Лимит — {_MAX_MESSAGE_LENGTH} символов.")
        return

    user_id = _participant_id(message)
    async with chat_locks.hold(user_id):
        session = await asyncio.to_thread(storage.load_active_session, user_id)
        if session is None:
            await message.answer("Наберите /start, чтобы начать опрос.")
            return
        try:
            async with llm_semaphore:
                reply, session = await asyncio.to_thread(advance, session, message.text, generator)
            if session.declined:
                # Отказ не оставляет pre-consent запись и не удаляет прошлые интервью.
                await asyncio.to_thread(storage.delete_session, user_id, session.interview_id)
            else:
                await asyncio.to_thread(storage.save_session, user_id, session)
        except ConcurrentSessionUpdate:
            logger.warning("Конкурентное обновление interview_id=%s", session.interview_id)
            await message.answer(
                "Предыдущий ответ ещё обрабатывается. Отправьте этот ответ ещё раз."
            )
            return
        except Exception:
            logger.exception("advance() упал для user_id=%s", user_id)
            await message.answer(_ERROR_REPLY)
            return
    await message.answer(reply)


async def main() -> None:
    config.validate(require_bot_secrets=True)
    await asyncio.to_thread(storage.healthcheck)
    bot = Bot(token=config.TELEGRAM_TOKEN)
    try:
        await dp.start_polling(bot, close_bot_session=False)
    finally:
        await bot.session.close()
        await asyncio.to_thread(generator.close)
        await asyncio.to_thread(storage.close)


if __name__ == "__main__":
    asyncio.run(main())
