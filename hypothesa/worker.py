"""Постоянный воркер очереди суммаризации.

Несколько экземпляров безопасно работают с одной БД: Storage забирает задачи
атомарно через SELECT ... FOR UPDATE SKIP LOCKED.
"""

from __future__ import annotations

import logging
import signal
import threading

from . import config
from .batch import run_pending_summaries
from .storage import Storage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("hypothesa.worker")


def main() -> None:
    config.validate()
    stop = threading.Event()

    def request_stop(signum, _frame) -> None:
        logger.info("Получен сигнал %s, завершаю воркер.", signum)
        stop.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    storage = Storage()
    try:
        storage.healthcheck()
        logger.info(
            "Воркер запущен: batch_limit=%s, poll_seconds=%s",
            config.WORKER_BATCH_LIMIT,
            config.WORKER_POLL_SECONDS,
        )
        while not stop.is_set():
            try:
                result = run_pending_summaries(
                    storage,
                    limit=config.WORKER_BATCH_LIMIT,
                )
                if result.processed or result.failed:
                    logger.info(
                        "Обработано: %s, ошибок: %s",
                        len(result.processed),
                        len(result.failed),
                    )
            except Exception:
                logger.exception("Ошибка цикла воркера; повторю попытку после паузы.")
            stop.wait(config.WORKER_POLL_SECONDS)
    finally:
        storage.close()
        logger.info("Воркер остановлен.")


if __name__ == "__main__":
    main()
