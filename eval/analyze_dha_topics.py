"""Прогнать сырые открытые ответы DHA_hackathon через BERTopic Hypothesa.

Старые GigaChat-сводки намеренно не используются: в них встречаются
галлюцинации и ответы модели вместо аспектов респондента.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from eval.preview_topics import build_preview_report, print_report, write_report
from hypothesa import config
from hypothesa.privacy import redact_pii
from hypothesa.summarize import is_stub_answer, prepare_answer_for_prompt
from hypothesa.topics import BERTopicBackend, TopicDocument

QUESTION_COLUMNS = {
    3: "user_ans_3",
    4: "user_ans_4",
}


def load_documents(
    source_path: Path,
) -> tuple[list[TopicDocument], list[dict], int]:
    """Собрать документы из проблем и положительных аспектов без PII-полей."""
    with source_path.open(encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        columns = set(reader.fieldnames or [])
        rows = list(reader)

    missing_columns = set(QUESTION_COLUMNS.values()) - columns
    if missing_columns:
        raise ValueError(f"В CSV отсутствуют нужные столбцы: {', '.join(sorted(missing_columns))}.")

    documents = []
    excluded = []
    for row_index, row in enumerate(rows):
        interview_number = row_index + 1
        interview_id = uuid5(
            NAMESPACE_URL,
            f"dha-hackathon:interview:{interview_number}",
        )
        for question_id, column in QUESTION_COLUMNS.items():
            value = row[column]
            if value is None or not value.strip():
                excluded.append(
                    {
                        "row": interview_number,
                        "question_id": question_id,
                        "reason": "missing",
                    }
                )
                continue

            raw_text = str(value).strip()
            if is_stub_answer(raw_text):
                excluded.append(
                    {
                        "row": interview_number,
                        "question_id": question_id,
                        "reason": "stub",
                    }
                )
                continue

            text = redact_pii(prepare_answer_for_prompt(raw_text))
            meaningful_tokens = [
                token for token in text.split() if any(char.isalpha() for char in token)
            ]
            if len(meaningful_tokens) < 2:
                excluded.append(
                    {
                        "row": interview_number,
                        "question_id": question_id,
                        "reason": "too_short",
                    }
                )
                continue

            document_id = uuid5(
                NAMESPACE_URL,
                (f"dha-hackathon:{interview_number}:{question_id}:{text.casefold()}"),
            )
            documents.append(
                TopicDocument(
                    document_id=document_id,
                    interview_id=interview_id,
                    question_id=question_id,
                    item_index=0,
                    text=text,
                )
            )
    return documents, excluded, len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/results/dha-topic-preview-raw.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    documents, excluded, csv_rows = load_documents(args.input)
    interview_count = len({document.interview_id for document in documents})
    print(
        f"Подготовлено {len(documents)} документов из "
        f"{interview_count} непустых интервью; исключено {len(excluded)} ответов.",
        flush=True,
    )

    backend = BERTopicBackend(
        embedding_model=config.EMBEDDING_MODEL,
        random_state=config.TOPIC_RANDOM_STATE,
        min_topic_size=config.TOPIC_MIN_SIZE,
        device=config.EMBEDDING_DEVICE,
        umap_n_neighbors=config.TOPIC_UMAP_NEIGHBORS,
        hdbscan_min_samples=config.TOPIC_MIN_SAMPLES,
        cluster_selection_method=config.TOPIC_CLUSTER_SELECTION_METHOD,
    )
    discovery = backend.fit(documents)
    report = build_preview_report(
        survey_id="dha-hackathon-raw-2026-07",
        documents=documents,
        discovery=discovery,
        embedding_model=config.EMBEDDING_MODEL,
        examples_per_topic=10,
        noise_examples=20,
    )
    report["source"] = {
        "path": str(args.input),
        "csv_rows": csv_rows,
        "included_documents": len(documents),
        "question_3_documents": sum(document.question_id == 3 for document in documents),
        "question_4_documents": sum(document.question_id == 4 for document in documents),
        "excluded_responses": excluded,
        "input": ("raw open answers only; age, city, and legacy GigaChat summaries excluded"),
    }
    report["topic_parameters"] = {
        "random_state": config.TOPIC_RANDOM_STATE,
        "min_topic_size": config.TOPIC_MIN_SIZE,
        "umap_n_neighbors": config.TOPIC_UMAP_NEIGHBORS,
        "hdbscan_min_samples": config.TOPIC_MIN_SAMPLES,
        "cluster_selection_method": config.TOPIC_CLUSTER_SELECTION_METHOD,
    }

    output_path = write_report(report, args.output)
    print_report(report)
    print(f"\nОтчёт: {output_path.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
