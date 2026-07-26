# Разработка Hypothesa

## Быстрый старт

1. Скопировать `.env.example` в `.env` и заполнить `TELEGRAM_TOKEN` и
   `PARTICIPANT_SALT`.
2. Запустить инфраструктуру: `docker compose up -d postgres ollama`.
3. Загрузить модели и применить миграции:
   `docker compose up ollama-init migrate`.
4. Проверить конфигурацию, миграции и модели: `python -m hypothesa.healthcheck`.
   Для машинной диагностики доступен `python -m hypothesa.healthcheck --json`.
5. Запустить бота: `python bot.py`.

Полный контейнерный запуск: `docker compose up --build bot`.

## Проверки

```powershell
python -m pytest -q
python -m ruff check .
python -m compileall -q .
alembic check
```

PostgreSQL integration-тест запускается только с отдельной тестовой БД:

```powershell
$env:TEST_DATABASE_URL='postgresql+psycopg://postgres:postgres@localhost:5432/hypothesa_test'
python -m pytest -m integration -q
```

Тот же набор автоматически выполняется GitHub Actions с отдельным PostgreSQL.

## Последовательность обработки данных

1. Бот сохраняет versioned-сессии и append-only события.
2. `python run_batch_summarization.py` формирует structured summaries и
   проверяет faithfulness отдельной моделью.
3. `python -m eval.run_eval` создаёт отдельный отчёт в `eval/results`.
   Baseline обновляется только после ручной проверки командой
   `python -m eval.run_eval --update-baseline`.
4. После накопления достаточного корпуса запускается
   `python run_topic_modeling.py`: BERTopic, стабильные ID, рейтинг и новая
   версия адаптивной анкеты.
5. `python run_experiment_report.py` показывает метрики control/adaptive и
   сообщает, достигнут ли минимальный размер выборки.

## Автоматический offline-конвейер

После применения миграций один проход можно проверить вручную:

```powershell
python run_scheduler.py --once
```

Постоянный scheduler запускается отдельным Docker-профилем:

```powershell
docker compose --profile automation up -d scheduler
docker compose logs -f scheduler
```

По умолчанию он проверяет очередь каждые 30 минут, но выполняет offline-фазу
только с 02:00 до 05:00 по `Europe/Moscow` и при отсутствии активных интервью.
Активной для этой проверки считается сессия, обновлённая за последние 30 минут.
Первый BERTopic-запуск происходит после 50 `faithful` интервью, последующие —
после 10 новых записей и не чаще раза в 24 часа. Если выбранные `topic_id` не
изменились, новый `topic_run` сохраняется, но дубликат анкеты не создаётся.

Когда scheduler включён, отдельно запускать `run_batch_summarization.py` и
`run_topic_modeling.py` не требуется.
Очистка записей старше `DATA_RETENTION_DAYS` также выполняется после каждого
успешного входа scheduler в maintenance window.

## Правила данных

- Telegram user id преобразуется HMAC-SHA256 до записи в PostgreSQL.
- Отказ от согласия переводит сессию в `abandoned`; ответы не собираются и
  сессия не попадает в batch.
- Участник может удалить индивидуальные данные командой `/delete_data`;
  плановая очистка выполняется scheduler или вручную `python run_data_retention.py`.
- Отказ от согласия удаляет только текущую pre-consent сессию, не затрагивая
  прошлые интервью участника.
- В BERTopic попадают только базовые открытые вопросы. Ответы на adaptive-вопросы
  исключены из discovery-корпуса, а email, телефоны и длинные идентификаторы маскируются.
- A/B-отчёт начинается с первой adaptive-анкеты и считает участника один раз.
- Непрошедшие faithfulness ответы сохраняются для аудита, но не попадают в
  тематический корпус.
- Завершённые сессии не удаляются: batch меняет их статус идемпотентно.
- Дашборд намеренно не включён до прохождения A/B-гейта на реальных данных.
