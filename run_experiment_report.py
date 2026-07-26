"""Рассчитать completion, длительность и A/B readiness по сохранённым сессиям."""

from __future__ import annotations

import json

from hypothesa import config
from hypothesa.metrics import build_experiment_report
from hypothesa.storage import Storage


def main() -> None:
    config.validate()
    storage = Storage()
    try:
        records = storage.list_experiment_records(config.SURVEY_ID)
        report = build_experiment_report(config.SURVEY_ID, records)
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    finally:
        storage.close()


if __name__ == "__main__":
    main()
