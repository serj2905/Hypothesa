"""Суммаризировать все интервью, накопленные ботом с момента прошлого запуска.

Судья (JUDGE_MODEL, другое семейство модели) грузится в VRAM только на время
этого запуска — отдельно от live-бота (см. hypothesa/batch.py, sprint_1.md).

Запуск:
    python run_batch_summarization.py
"""

from hypothesa import config
from hypothesa.batch import run_pending_summaries
from hypothesa.storage import Storage


def main() -> None:
    config.validate()
    storage = Storage()
    try:
        result = run_pending_summaries(storage)
        print(f"Суммаризировано интервью: {len(result.processed)}")
        print(f"Ошибок: {len(result.failed)}")
    finally:
        storage.close()


if __name__ == "__main__":
    main()
