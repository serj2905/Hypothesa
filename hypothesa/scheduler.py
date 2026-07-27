"""Периодический offline-scheduler для тематического обновления."""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from . import config
from .automation import run_automatic_pipeline_once
from .storage import Storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("hypothesa.scheduler")


def within_maintenance_window(now: datetime) -> bool:
    local = now.astimezone(ZoneInfo(config.AUTOMATION_TIMEZONE))
    elapsed_hours = (local.hour + local.minute / 60 - config.AUTOMATION_WINDOW_START_HOUR) % 24
    return elapsed_hours < config.AUTOMATION_WINDOW_HOURS


def run_once(storage: Storage) -> None:
    result = run_automatic_pipeline_once(storage)
    logger.info(
        "automation_result=%s",
        json.dumps(result.as_dict(), ensure_ascii=False, default=str),
    )
    purged = storage.purge_expired_data(config.DATA_RETENTION_DAYS)
    logger.info("retention_purged=%s", purged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--once",
        action="store_true",
        help="Выполнить один проход сейчас, игнорируя временное окно.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config.validate()
    storage = Storage()
    try:
        if args.once:
            run_once(storage)
            return

        logger.info(
            "Scheduler запущен: interval=%s мин, window=%02d:00 +%s ч, timezone=%s",
            config.AUTOMATION_INTERVAL_MINUTES,
            config.AUTOMATION_WINDOW_START_HOUR,
            config.AUTOMATION_WINDOW_HOURS,
            config.AUTOMATION_TIMEZONE,
        )
        while True:
            try:
                now = datetime.now(tz=ZoneInfo(config.AUTOMATION_TIMEZONE))
                if within_maintenance_window(now):
                    run_once(storage)
            except Exception:
                logger.exception("Автоматический конвейер завершился ошибкой.")
            time.sleep(config.AUTOMATION_INTERVAL_MINUTES * 60)
    except KeyboardInterrupt:
        logger.info("Scheduler остановлен.")
    finally:
        storage.close()


if __name__ == "__main__":
    main()
