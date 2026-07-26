"""Удалить индивидуальные данные старше настроенного срока хранения."""

from hypothesa import config
from hypothesa.storage import Storage


def main() -> None:
    config.validate()
    storage = Storage()
    try:
        deleted = storage.purge_expired_data(config.DATA_RETENTION_DAYS)
        print(f"Удалено интервью: {deleted}")
    finally:
        storage.close()


if __name__ == "__main__":
    main()
