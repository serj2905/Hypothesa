"""Тематический цикл: BERTopic → стабильные темы → рейтинг → новая анкета."""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import numpy as np

from .interview import DEFAULT_QUESTIONS, QuestionSpec
from .metrics import wilson_interval


class InsufficientDocuments(ValueError):
    """Для устойчивой кластеризации недостаточно документов."""


RUSSIAN_TOPIC_STOP_WORDS = frozenset(
    """
    а без более больше был была были было быть в вам вас ведь весь во вот все
    всего всех вы где да даже для до его ее если есть еще же за здесь и из или
    им их к как ко когда кто ли либо мне может мой моя мое мои мы на над надо
    наш наша наше наши не него нее нет ни но ну о об однако он она они оно от
    очень по под при про раз с сам сама сами со так такой также там те тем то
    того тоже тут ты у уже что чтобы эта эти это я
    банк банка банке банком сбер сбера сбербанк сбербанка
    """.split()  # noqa: SIM905 — такой список проще проверять и дополнять.
)


def build_topic_vectorizer():
    """Создать c-TF-IDF vectorizer без служебных русских и доменных слов."""
    from sklearn.feature_extraction.text import CountVectorizer

    return CountVectorizer(
        stop_words=sorted(RUSSIAN_TOPIC_STOP_WORDS),
        ngram_range=(1, 2),
    )


@dataclass(frozen=True)
class TopicDocument:
    document_id: UUID
    interview_id: UUID
    question_id: int
    item_index: int
    text: str


@dataclass(frozen=True)
class DiscoveredTopic:
    local_id: int
    label: str
    keywords: list[str]
    centroid: list[float]


@dataclass(frozen=True)
class LocalAssignment:
    document_id: UUID
    local_topic_id: int | None
    probability: float | None = None


@dataclass(frozen=True)
class TopicDiscovery:
    topics: list[DiscoveredTopic]
    assignments: list[LocalAssignment]
    metrics: dict[str, float | None] = field(default_factory=dict)


@dataclass(frozen=True)
class RegisteredTopic:
    topic_id: UUID
    survey_id: str
    label: str
    keywords: list[str]
    centroid: list[float]
    rating: int = 0
    mention_count: int = 0
    active: bool = True
    # Производные величины: заполняются в calculate_ratings и не хранятся в БД,
    # потому что однозначно восстанавливаются из mention_count и размера выборки.
    interview_count: int = 0
    prevalence_ci95: tuple[float, float] = (0.0, 0.0)


@dataclass(frozen=True)
class TopicAssignment:
    document_id: UUID
    topic_id: UUID | None
    probability: float | None = None


@dataclass(frozen=True)
class QuestionnaireVersion:
    survey_id: str
    version: int
    questions: list[QuestionSpec]
    source_topic_ids: list[UUID]


@dataclass(frozen=True)
class TopicCycleResult:
    run_id: UUID
    started_at: datetime
    completed_at: datetime
    document_count: int
    topics: list[RegisteredTopic]
    assignments: list[TopicAssignment]
    questionnaire: QuestionnaireVersion
    questionnaire_changed: bool
    metrics: dict[str, float | None]


class TopicBackend(Protocol):
    def fit(self, documents: Sequence[TopicDocument]) -> TopicDiscovery: ...


class TopicStore(Protocol):
    def list_topic_documents(self, survey_id: str) -> list[TopicDocument]: ...

    def load_topics(self, survey_id: str) -> list[RegisteredTopic]: ...

    def next_questionnaire_version(self, survey_id: str) -> int: ...

    def load_latest_questionnaire(self, survey_id: str) -> QuestionnaireVersion | None: ...

    def save_topic_cycle(self, result: TopicCycleResult) -> None: ...


class BERTopicBackend:
    """Ленивый production-адаптер; тяжёлые ML-зависимости не нужны боту."""

    def __init__(
        self,
        *,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        random_state: int = 42,
        min_topic_size: int = 5,
        device: str = "cpu",
        embedding_prefix: str = "",
        umap_n_neighbors: int = 15,
        hdbscan_min_samples: int | None = None,
        cluster_selection_method: Literal["eom", "leaf"] = "eom",
    ) -> None:
        self.embedding_model = embedding_model
        self.random_state = random_state
        self.min_topic_size = min_topic_size
        self.device = device
        self.embedding_prefix = embedding_prefix
        self.umap_n_neighbors = umap_n_neighbors
        self.hdbscan_min_samples = hdbscan_min_samples
        self.cluster_selection_method = cluster_selection_method

    def encode_documents(self, documents: Sequence[TopicDocument]) -> np.ndarray:
        """Один раз посчитать нормализованные embeddings для fit или eval-сетки."""
        from sentence_transformers import SentenceTransformer

        texts = [f"{self.embedding_prefix}{document.text}" for document in documents]
        encoder = SentenceTransformer(self.embedding_model, device=self.device)
        return np.asarray(encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False))

    def fit(
        self,
        documents: Sequence[TopicDocument],
        *,
        embeddings: np.ndarray | None = None,
    ) -> TopicDiscovery:
        if len(documents) < max(5, self.min_topic_size):
            raise InsufficientDocuments(
                f"Нужно минимум {max(5, self.min_topic_size)} документов, получено {len(documents)}."
            )

        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from sklearn.metrics import silhouette_score
        from umap import UMAP

        texts = [document.text for document in documents]
        embeddings = (
            self.encode_documents(documents) if embeddings is None else np.asarray(embeddings)
        )
        if embeddings.ndim != 2 or embeddings.shape[0] != len(documents):
            raise ValueError("Embeddings должны быть матрицей с одной строкой на документ.")
        n_neighbors = min(self.umap_n_neighbors, len(documents) - 1)
        n_components = min(5, len(documents) - 2)
        umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=0.0,
            metric="cosine",
            random_state=self.random_state,
        )
        cluster_model = HDBSCAN(
            min_cluster_size=self.min_topic_size,
            min_samples=self.hdbscan_min_samples,
            metric="euclidean",
            cluster_selection_method=self.cluster_selection_method,
            prediction_data=True,
        )
        model = BERTopic(
            language="multilingual",
            embedding_model=None,
            umap_model=umap_model,
            hdbscan_model=cluster_model,
            vectorizer_model=build_topic_vectorizer(),
            calculate_probabilities=True,
            verbose=False,
        )
        local_ids, probabilities = model.fit_transform(texts, embeddings)

        discovered = []
        for local_id in sorted(set(local_ids) - {-1}):
            indexes = [index for index, value in enumerate(local_ids) if value == local_id]
            keywords = [word for word, _ in (model.get_topic(local_id) or [])[:8]]
            centroid = _normalized_centroid(embeddings[indexes])
            discovered.append(
                DiscoveredTopic(
                    local_id=local_id,
                    label=", ".join(keywords[:3]) or f"Тема {local_id}",
                    keywords=keywords,
                    centroid=centroid.tolist(),
                )
            )

        assignments = []
        for index, (document, local_id) in enumerate(zip(documents, local_ids, strict=True)):
            probability = None
            if probabilities is not None and local_id != -1:
                probability = float(np.max(np.asarray(probabilities[index])))
            assignments.append(
                LocalAssignment(
                    document_id=document.document_id,
                    local_topic_id=None if local_id == -1 else int(local_id),
                    probability=probability,
                )
            )

        non_noise = [value for value in local_ids if value != -1]
        silhouette = None
        if len(set(non_noise)) >= 2:
            mask = np.asarray(local_ids) != -1
            silhouette = float(silhouette_score(embeddings[mask], np.asarray(local_ids)[mask]))
        counts = Counter(local_ids)
        non_noise_counts = [count for topic, count in counts.most_common() if topic != -1]
        top_five = sum(non_noise_counts[:5])
        metrics = {
            "noise_ratio": local_ids.count(-1) / len(local_ids),
            "top5_coverage": top_five / len(local_ids),
            "silhouette": silhouette,
        }
        return TopicDiscovery(discovered, assignments, metrics)


def _normalized_centroid(vectors: np.ndarray) -> np.ndarray:
    centroid = np.mean(vectors, axis=0)
    norm = np.linalg.norm(centroid)
    return centroid / norm if norm else centroid


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_vector = np.asarray(left, dtype=float)
    right_vector = np.asarray(right, dtype=float)
    denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
    if denominator == 0 or left_vector.shape != right_vector.shape:
        return -1.0
    return float(np.dot(left_vector, right_vector) / denominator)


def reconcile_topics(
    survey_id: str,
    discovered: Sequence[DiscoveredTopic],
    existing: Sequence[RegisteredTopic],
    *,
    similarity_threshold: float = 0.82,
) -> tuple[list[RegisteredTopic], dict[int, UUID]]:
    """Сопоставить новые кластеры со старыми центроидами, сохраняя topic_id."""
    unmatched = {topic.topic_id: topic for topic in existing}
    registered = []
    local_to_stable = {}

    for topic in discovered:
        candidates = [(_cosine(topic.centroid, old.centroid), old) for old in unmatched.values()]
        similarity, matched = max(candidates, default=(-1.0, None), key=lambda item: item[0])
        if matched is not None and similarity >= similarity_threshold:
            topic_id = matched.topic_id
            rating = matched.rating
            unmatched.pop(topic_id)
        else:
            signature = "|".join(_slug(word) for word in topic.keywords[:5]) or _slug(topic.label)
            topic_id = uuid5(NAMESPACE_URL, f"hypothesa:{survey_id}:{signature}")
            rating = 0
        local_to_stable[topic.local_id] = topic_id
        registered.append(
            RegisteredTopic(
                topic_id=topic_id,
                survey_id=survey_id,
                label=topic.label,
                keywords=topic.keywords,
                centroid=topic.centroid,
                rating=rating,
            )
        )

    # Старые темы не исчезают из реестра: отсутствие упоминаний понизит рейтинг.
    registered.extend(unmatched.values())
    return registered, local_to_stable


def _slug(value: str) -> str:
    return re.sub(r"[^\w]+", "-", value.lower().replace("ё", "е")).strip("-")


def stabilize_assignments(
    assignments: Sequence[LocalAssignment],
    local_to_stable: dict[int, UUID],
) -> list[TopicAssignment]:
    return [
        TopicAssignment(
            document_id=item.document_id,
            topic_id=local_to_stable.get(item.local_topic_id)
            if item.local_topic_id is not None
            else None,
            probability=item.probability,
        )
        for item in assignments
    ]


def calculate_ratings(
    topics: Sequence[RegisteredTopic],
    assignments: Sequence[TopicAssignment],
    documents: Sequence[TopicDocument],
    *,
    min_mentions: int = 2,
    min_prevalence: float = 0.05,
) -> list[RegisteredTopic]:
    """Оценить распространённость темы со сглаживанием без double count.

    `rating` — точечная оценка со сглаживанием Лапласа, `prevalence_ci95` —
    интервал Уилсона по сырой доле упоминаний. Оба говорят об одной величине,
    но интервал показывает, насколько на этой выборке вообще можно доверять
    порядку тем: на пилотных n он шире, чем расстояние между соседними темами.
    """
    if min_mentions < 1:
        raise ValueError("min_mentions должен быть положительным.")
    if not 0 <= min_prevalence <= 1:
        raise ValueError("min_prevalence должен находиться между 0 и 1.")
    document_to_interview = {document.document_id: document.interview_id for document in documents}
    interviews = set(document_to_interview.values())
    mentions: dict[UUID, set[UUID]] = defaultdict(set)
    for assignment in assignments:
        if assignment.topic_id is not None:
            mentions[assignment.topic_id].add(document_to_interview[assignment.document_id])

    rated = []
    for topic in topics:
        topic_mentions = mentions[topic.topic_id]
        mention_count = len(topic_mentions)
        # Laplace smoothing не даёт малой выборке создавать экстремальные 0/100.
        prevalence = (mention_count + 1) / (len(interviews) + 2)
        rating = round(prevalence * 100)
        rated.append(
            RegisteredTopic(
                topic_id=topic.topic_id,
                survey_id=topic.survey_id,
                label=topic.label,
                keywords=topic.keywords,
                centroid=topic.centroid,
                rating=rating,
                mention_count=mention_count,
                active=(mention_count >= min_mentions and prevalence >= min_prevalence),
                interview_count=len(interviews),
                prevalence_ci95=wilson_interval(mention_count, len(interviews)),
            )
        )
    return rated


def build_questionnaire(
    survey_id: str,
    version: int,
    topics: Sequence[RegisteredTopic],
    *,
    base_questions: Sequence[QuestionSpec] = DEFAULT_QUESTIONS,
    topic_question_count: int = 2,
    seed: int = 42,
) -> QuestionnaireVersion:
    """Взвешенно выбрать активные темы без повторов и создать версию анкеты."""
    candidates = [topic for topic in topics if topic.active and topic.rating > 0]
    rng = random.Random(seed)
    ranked = sorted(
        candidates,
        key=lambda topic: rng.random() ** (1 / topic.rating),
        reverse=True,
    )[:topic_question_count]
    questions = [question.model_copy(deep=True) for question in base_questions]
    next_id = max((question.id for question in questions), default=0) + 1
    for offset, topic in enumerate(ranked):
        questions.append(
            QuestionSpec(
                id=next_id + offset,
                kind="open",
                max_followups=1,
                topic_id=topic.topic_id,
                text=(
                    f"Расскажите о вашем опыте, связанном с темой «{topic.label}». "
                    "Что именно было для вас важно?"
                ),
            )
        )
    return QuestionnaireVersion(
        survey_id=survey_id,
        version=version,
        questions=questions,
        source_topic_ids=[topic.topic_id for topic in ranked],
    )


def run_topic_cycle(
    store: TopicStore,
    survey_id: str,
    backend: TopicBackend,
    *,
    similarity_threshold: float = 0.82,
    min_mentions: int = 2,
    min_prevalence: float = 0.05,
    topic_question_count: int = 2,
    seed: int = 42,
) -> TopicCycleResult:
    started_at = datetime.now(UTC)
    documents = store.list_topic_documents(survey_id)
    discovery = backend.fit(documents)
    topics, mapping = reconcile_topics(
        survey_id,
        discovery.topics,
        store.load_topics(survey_id),
        similarity_threshold=similarity_threshold,
    )
    assignments = stabilize_assignments(discovery.assignments, mapping)
    rated_topics = calculate_ratings(
        topics,
        assignments,
        documents,
        min_mentions=min_mentions,
        min_prevalence=min_prevalence,
    )
    candidate_questionnaire = build_questionnaire(
        survey_id,
        store.next_questionnaire_version(survey_id),
        rated_topics,
        topic_question_count=topic_question_count,
        seed=seed,
    )
    latest_questionnaire = store.load_latest_questionnaire(survey_id)
    questionnaire_changed = (
        latest_questionnaire is None
        or latest_questionnaire.source_topic_ids != candidate_questionnaire.source_topic_ids
    )
    questionnaire = candidate_questionnaire if questionnaire_changed else latest_questionnaire
    completed_at = datetime.now(UTC)
    metrics = dict(discovery.metrics)
    metrics["runtime_seconds"] = (completed_at - started_at).total_seconds()
    result = TopicCycleResult(
        run_id=uuid4(),
        started_at=started_at,
        completed_at=completed_at,
        document_count=len(documents),
        topics=rated_topics,
        assignments=assignments,
        questionnaire=questionnaire,
        questionnaire_changed=questionnaire_changed,
        metrics=metrics,
    )
    store.save_topic_cycle(result)
    return result
