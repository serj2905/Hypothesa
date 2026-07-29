"""Read-only BERTopic preview на текущем корпусе PostgreSQL.

Команда намеренно не вызывает ``run_topic_cycle`` и ``save_topic_cycle``:
она только читает обезличенные документы, запускает backend и сохраняет
локальный диагностический отчёт в ``eval/results``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import UUID

from hypothesa import config
from hypothesa.metrics import wilson_interval
from hypothesa.storage import Storage
from hypothesa.topics import (
    BERTopicBackend,
    InsufficientDocuments,
    TopicBackend,
    TopicDiscovery,
    TopicDocument,
)

RESULTS_DIR = Path(__file__).parent / "results"


class TopicDocumentSource(Protocol):
    def list_topic_documents(self, survey_id: str) -> list[TopicDocument]: ...


def _topic_interviews(
    documents_by_id: dict[UUID, TopicDocument],
    discovery: TopicDiscovery,
) -> dict[int, set[UUID]]:
    """Сгруппировать интервью по темам так же, как это делает calculate_ratings.

    Упоминание считается один раз на интервью, поэтому многословный респондент
    не поднимает распространённость темы в одиночку. Ключ -1 — шум.
    """
    grouped: dict[int, set[UUID]] = defaultdict(set)
    for assignment in discovery.assignments:
        document = documents_by_id.get(assignment.document_id)
        if document is None:
            raise ValueError(
                f"Назначение ссылается на неизвестный документ {assignment.document_id}."
            )
        key = -1 if assignment.local_topic_id is None else assignment.local_topic_id
        grouped[key].add(document.interview_id)
    return grouped


def _ranked_examples(
    documents_by_id: dict[UUID, TopicDocument],
    discovery: TopicDiscovery,
    local_topic_id: int | None,
    *,
    limit: int,
) -> list[dict]:
    candidates = []
    for assignment in discovery.assignments:
        if assignment.local_topic_id != local_topic_id:
            continue
        document = documents_by_id.get(assignment.document_id)
        if document is None:
            raise ValueError(
                f"Назначение ссылается на неизвестный документ {assignment.document_id}."
            )
        candidates.append(
            (
                -1.0 if assignment.probability is None else assignment.probability,
                document,
                assignment.probability,
            )
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "text": document.text,
            "question_id": document.question_id,
            "probability": probability,
        }
        for _, document, probability in candidates[:limit]
    ]


def build_preview_report(
    *,
    survey_id: str,
    documents: Sequence[TopicDocument],
    discovery: TopicDiscovery,
    embedding_model: str,
    examples_per_topic: int = 5,
    noise_examples: int = 10,
) -> dict:
    """Собрать компактный отчёт без центроидов и полного корпуса."""
    documents_by_id = {document.document_id: document for document in documents}
    assignment_counts = Counter(
        -1 if assignment.local_topic_id is None else assignment.local_topic_id
        for assignment in discovery.assignments
    )
    topic_interviews = _topic_interviews(documents_by_id, discovery)
    interview_count = len({document.interview_id for document in documents})
    topics = [
        {
            "local_id": topic.local_id,
            "label": topic.label,
            "keywords": topic.keywords,
            "document_count": assignment_counts[topic.local_id],
            "interview_count": len(topic_interviews[topic.local_id]),
            "prevalence": (
                len(topic_interviews[topic.local_id]) / interview_count if interview_count else 0.0
            ),
            "prevalence_ci95": wilson_interval(
                len(topic_interviews[topic.local_id]),
                interview_count,
            ),
            "examples": _ranked_examples(
                documents_by_id,
                discovery,
                topic.local_id,
                limit=examples_per_topic,
            ),
        }
        for topic in discovery.topics
    ]
    # Порядок по охвату интервью повторяет ранжирование calculate_ratings,
    # поэтому preview показывает тот же топ, что построил бы реальный цикл.
    topics.sort(
        key=lambda topic: (topic["interview_count"], topic["document_count"]),
        reverse=True,
    )
    return {
        "status": "ok",
        "timestamp": datetime.now(UTC).isoformat(),
        "survey_id": survey_id,
        "embedding_model": embedding_model,
        "document_count": len(documents),
        "interview_count": interview_count,
        "topic_count": len(topics),
        "noise_count": assignment_counts[-1],
        "metrics": discovery.metrics,
        "topics": topics,
        "noise_examples": _ranked_examples(
            documents_by_id,
            discovery,
            None,
            limit=noise_examples,
        ),
    }


def preview_topics(
    source: TopicDocumentSource,
    backend: TopicBackend,
    *,
    survey_id: str,
    embedding_model: str,
    examples_per_topic: int = 5,
    noise_examples: int = 10,
) -> dict:
    """Прочитать документы и построить preview без любых вызовов записи."""
    documents = source.list_topic_documents(survey_id)
    discovery = backend.fit(documents)
    return build_preview_report(
        survey_id=survey_id,
        documents=documents,
        discovery=discovery,
        embedding_model=embedding_model,
        examples_per_topic=examples_per_topic,
        noise_examples=noise_examples,
    )


def write_report(report: dict, output_path: Path | None = None) -> Path:
    if output_path is None:
        RESULTS_DIR.mkdir(exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output_path = RESULTS_DIR / f"topic-preview-{stamp}.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def print_report(report: dict) -> None:
    metrics = report["metrics"]
    silhouette = metrics.get("silhouette")
    silhouette_text = "—" if silhouette is None else f"{silhouette:.3f}"
    coverage = metrics.get("top5_coverage")
    coverage_text = "—" if coverage is None else f"{coverage:.1%}"
    print(
        f"BERTopic preview: survey={report['survey_id']}, "
        f"documents={report['document_count']}, interviews={report['interview_count']}"
    )
    print(
        f"Темы: {report['topic_count']}; шум: {report['noise_count']} "
        f"({metrics.get('noise_ratio', 0.0):.1%}); покрытие топ-5: {coverage_text}; "
        f"silhouette: {silhouette_text}"
    )
    print("Распространённость — доля интервью с упоминанием темы, в скобках 95% ДИ Уилсона.")
    for topic in report["topics"]:
        low, high = topic["prevalence_ci95"]
        print(
            f"\n[{topic['local_id']}] {topic['label']} "
            f"({topic['document_count']} документов; "
            f"{topic['interview_count']} из {report['interview_count']} интервью — "
            f"{topic['prevalence']:.0%}, 95% ДИ {low:.0%}–{high:.0%})"
        )
        print(f"    keywords: {', '.join(topic['keywords'])}")
        for example in topic["examples"]:
            probability = example["probability"]
            probability_text = "—" if probability is None else f"{probability:.3f}"
            print(f"    - [{probability_text}] {example['text']}")
    if report["noise_examples"]:
        print("\nПримеры шума:")
        for example in report["noise_examples"]:
            print(f"    - {example['text']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey-id", default=config.SURVEY_ID)
    parser.add_argument("--examples-per-topic", type=int, default=5)
    parser.add_argument("--noise-examples", type=int, default=10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.examples_per_topic < 1 or args.noise_examples < 0:
        raise SystemExit("Лимиты примеров должны быть неотрицательными.")

    backend = BERTopicBackend(
        embedding_model=config.EMBEDDING_MODEL,
        random_state=config.TOPIC_RANDOM_STATE,
        min_topic_size=config.TOPIC_MIN_SIZE,
        device=config.EMBEDDING_DEVICE,
        umap_n_neighbors=config.TOPIC_UMAP_NEIGHBORS,
        hdbscan_min_samples=config.TOPIC_MIN_SAMPLES,
        cluster_selection_method=config.TOPIC_CLUSTER_SELECTION_METHOD,
    )
    storage = Storage()
    try:
        try:
            report = preview_topics(
                storage,
                backend,
                survey_id=args.survey_id,
                embedding_model=config.EMBEDDING_MODEL,
                examples_per_topic=args.examples_per_topic,
                noise_examples=args.noise_examples,
            )
        except InsufficientDocuments as exc:
            print(f"BERTopic preview не запущен: {exc}")
            return 2
    finally:
        storage.close()

    output_path = write_report(report, args.output)
    print_report(report)
    print(f"\nОтчёт: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
