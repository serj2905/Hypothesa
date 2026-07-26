"""Построить темы, рейтинг и следующую адаптивную версию анкеты."""

from __future__ import annotations

from hypothesa import config
from hypothesa.storage import Storage
from hypothesa.topics import BERTopicBackend, run_topic_cycle


def main() -> None:
    config.validate()
    storage = Storage()
    try:
        backend = BERTopicBackend(
            embedding_model=config.EMBEDDING_MODEL,
            random_state=config.TOPIC_RANDOM_STATE,
            min_topic_size=config.TOPIC_MIN_SIZE,
            device=config.EMBEDDING_DEVICE,
        )
        result = run_topic_cycle(
            storage,
            config.SURVEY_ID,
            backend,
            similarity_threshold=config.TOPIC_SIMILARITY_THRESHOLD,
            min_mentions=config.TOPIC_MIN_MENTIONS,
            min_prevalence=config.TOPIC_MIN_PREVALENCE,
            topic_question_count=config.TOPIC_QUESTION_COUNT,
            seed=config.TOPIC_RANDOM_STATE,
        )
        print(f"Документов: {result.document_count}")
        print(f"Тем: {len(result.topics)}")
        print(f"Новая версия анкеты: {result.questionnaire.version}")
        print(f"Анкета изменилась: {result.questionnaire_changed}")
        print(f"Метрики: {result.metrics}")
    finally:
        storage.close()


if __name__ == "__main__":
    main()
