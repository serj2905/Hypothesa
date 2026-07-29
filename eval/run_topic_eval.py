"""Сравнение BERTopic-конфигураций на размеченном аспектном smoke-корпусе."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from time import perf_counter
from uuid import NAMESPACE_URL, uuid5

from sklearn.metrics import adjusted_rand_score

from eval.topic_cases import TOPIC_CASES
from hypothesa.topics import BERTopicBackend, TopicDocument

RESULTS_DIR = Path(__file__).parent / "results"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str
    embedding_prefix: str = ""


@dataclass(frozen=True)
class ClusterSpec:
    n_neighbors: int
    min_topic_size: int
    min_samples: int | None
    selection: str

    @property
    def name(self) -> str:
        samples = "default" if self.min_samples is None else str(self.min_samples)
        return (
            f"nn={self.n_neighbors},size={self.min_topic_size},samples={samples},{self.selection}"
        )


MODELS = {
    "minilm": ModelSpec(
        name="minilm",
        model_id="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ),
    "e5": ModelSpec(
        name="e5",
        model_id="intfloat/multilingual-e5-base",
        embedding_prefix="passage: ",
    ),
}


def build_documents() -> tuple[list[TopicDocument], list[int], list[str]]:
    documents: list[TopicDocument] = []
    expected: list[int] = []
    labels = list(TOPIC_CASES)
    for label_id, (label, texts) in enumerate(TOPIC_CASES.items()):
        for index, text in enumerate(texts):
            documents.append(
                TopicDocument(
                    document_id=uuid5(NAMESPACE_URL, f"topic-eval:{label}:{index}"),
                    interview_id=uuid5(
                        NAMESPACE_URL,
                        f"topic-eval-interview:{label}:{index // 2}",
                    ),
                    question_id=3 if index % 2 == 0 else 4,
                    item_index=index % 2,
                    text=text,
                )
            )
            expected.append(label_id)
    return documents, expected, labels


def cluster_specs(*, quick: bool) -> list[ClusterSpec]:
    if quick:
        return [
            ClusterSpec(15, 5, None, "eom"),
            ClusterSpec(10, 5, 1, "leaf"),
            ClusterSpec(5, 5, 2, "eom"),
        ]
    return [
        ClusterSpec(neighbors, size, samples, selection)
        for neighbors, size, samples, selection in product(
            (5, 10, 15),
            (3, 5),
            (1, 2, None),
            ("eom", "leaf"),
        )
    ]


def cluster_purity(expected: list[int], predicted: list[int]) -> float:
    members: dict[int, list[int]] = {}
    for truth, cluster in zip(expected, predicted, strict=True):
        if cluster != -1:
            members.setdefault(cluster, []).append(truth)
    assigned = sum(len(values) for values in members.values())
    if not assigned:
        return 0.0
    correct = sum(Counter(values).most_common(1)[0][1] for values in members.values())
    return correct / assigned


def evaluate_model(
    model: ModelSpec,
    specifications: list[ClusterSpec],
    documents: list[TopicDocument],
    expected: list[int],
    labels: list[str],
) -> list[dict]:
    encoder = BERTopicBackend(
        embedding_model=model.model_id,
        embedding_prefix=model.embedding_prefix,
    )
    started = perf_counter()
    embeddings = encoder.encode_documents(documents)
    embedding_seconds = perf_counter() - started

    rows = []
    for specification in specifications:
        backend = BERTopicBackend(
            embedding_model=model.model_id,
            embedding_prefix=model.embedding_prefix,
            min_topic_size=specification.min_topic_size,
            hdbscan_min_samples=specification.min_samples,
            umap_n_neighbors=specification.n_neighbors,
            cluster_selection_method=specification.selection,
        )
        started = perf_counter()
        discovery = backend.fit(documents, embeddings=embeddings)
        runtime = perf_counter() - started
        predicted = [
            -1 if assignment.local_topic_id is None else assignment.local_topic_id
            for assignment in discovery.assignments
        ]
        topics = []
        for topic in discovery.topics:
            assigned_truth = [
                labels[truth]
                for truth, cluster in zip(expected, predicted, strict=True)
                if cluster == topic.local_id
            ]
            topics.append(
                {
                    "local_id": topic.local_id,
                    "label": topic.label,
                    "keywords": topic.keywords,
                    "size": len(assigned_truth),
                    "expected_aspects": dict(Counter(assigned_truth)),
                }
            )
        noise_ratio = float(discovery.metrics["noise_ratio"] or 0.0)
        rows.append(
            {
                "model": model.name,
                "model_id": model.model_id,
                "embedding_seconds": embedding_seconds,
                "configuration": asdict(specification),
                "configuration_name": specification.name,
                "topic_count": len(discovery.topics),
                "noise_ratio": noise_ratio,
                "coverage": 1.0 - noise_ratio,
                "silhouette": discovery.metrics["silhouette"],
                "adjusted_rand": adjusted_rand_score(expected, predicted),
                "assigned_purity": cluster_purity(expected, predicted),
                "runtime_seconds": runtime,
                "topics": topics,
            }
        )
    return rows


def result_sort_key(row: dict) -> tuple[float, float, float, float]:
    return (
        row["adjusted_rand"],
        row["assigned_purity"],
        row["coverage"],
        row["silhouette"] if row["silhouette"] is not None else -1.0,
    )


def print_results(rows: list[dict], *, limit: int = 12) -> None:
    ranked = sorted(rows, key=result_sort_key, reverse=True)
    print(
        f"{'model':<7} {'configuration':<38} {'topics':>6} "
        f"{'noise':>7} {'ARI':>7} {'purity':>8} {'silh.':>7}"
    )
    print("-" * 91)
    for row in ranked[:limit]:
        silhouette = row["silhouette"]
        silhouette_text = "—" if silhouette is None else f"{silhouette:.3f}"
        print(
            f"{row['model']:<7} {row['configuration_name']:<38} "
            f"{row['topic_count']:>6} {row['noise_ratio']:>6.0%} "
            f"{row['adjusted_rand']:>7.3f} {row['assigned_purity']:>7.0%} "
            f"{silhouette_text:>7}"
        )

    print("\nЛучший вариант по каждой embedding-модели:")
    for model_name in sorted({row["model"] for row in rows}):
        best = max(
            (row for row in rows if row["model"] == model_name),
            key=result_sort_key,
        )
        print(f"\n{model_name}: {best['configuration_name']}")
        for topic in best["topics"]:
            print(
                f"  {topic['local_id']}: {topic['label']} ({topic['size']}) "
                f"{topic['expected_aspects']}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODELS),
        default=["minilm", "e5"],
        help="Embedding-модели для сравнения.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Проверить только baseline и два кандидатных набора параметров.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    documents, expected, labels = build_documents()
    specifications = cluster_specs(quick=args.quick)
    rows = []
    for model_name in args.models:
        rows.extend(
            evaluate_model(
                MODELS[model_name],
                specifications,
                documents,
                expected,
                labels,
            )
        )

    report = {
        "timestamp": datetime.now(UTC).isoformat(),
        "document_count": len(documents),
        "expected_aspects": labels,
        "results": rows,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = RESULTS_DIR / f"topic-eval-{stamp}.json"
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print_results(rows)
    print(f"\nПолный отчёт: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
