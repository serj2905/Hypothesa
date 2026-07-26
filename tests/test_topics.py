from uuid import uuid4

from hypothesa.topics import (
    DiscoveredTopic,
    LocalAssignment,
    RegisteredTopic,
    TopicDiscovery,
    TopicDocument,
    build_questionnaire,
    calculate_ratings,
    reconcile_topics,
    run_topic_cycle,
    stabilize_assignments,
)


def document(interview_id, index, text="аспект") -> TopicDocument:
    return TopicDocument(
        document_id=uuid4(),
        interview_id=interview_id,
        question_id=3,
        item_index=index,
        text=text,
    )


def test_reconcile_preserves_stable_id_for_similar_centroid() -> None:
    stable_id = uuid4()
    existing = RegisteredTopic(
        topic_id=stable_id,
        survey_id="survey",
        label="Старое имя",
        keywords=["комиссия"],
        centroid=[1.0, 0.0],
        rating=5,
    )
    discovered = DiscoveredTopic(
        local_id=7,
        label="Комиссии",
        keywords=["комиссия", "перевод"],
        centroid=[0.99, 0.01],
    )

    topics, mapping = reconcile_topics("survey", [discovered], [existing])

    assert mapping[7] == stable_id
    assert topics[0].topic_id == stable_id
    assert topics[0].rating == 5


def test_rating_is_recomputed_without_double_counting() -> None:
    first_interview, second_interview = uuid4(), uuid4()
    documents = [document(first_interview, 0), document(second_interview, 0)]
    first_topic = RegisteredTopic(
        topic_id=uuid4(),
        survey_id="survey",
        label="Комиссия",
        keywords=["комиссия"],
        centroid=[1.0, 0.0],
    )
    second_topic = RegisteredTopic(
        topic_id=uuid4(),
        survey_id="survey",
        label="Приложение",
        keywords=["приложение"],
        centroid=[0.0, 1.0],
    )
    assignments = stabilize_assignments(
        [
            LocalAssignment(documents[0].document_id, 1),
            LocalAssignment(documents[1].document_id, 1),
        ],
        {1: first_topic.topic_id},
    )

    rated = calculate_ratings([first_topic, second_topic], assignments, documents)
    values = {topic.topic_id: topic for topic in rated}

    assert values[first_topic.topic_id].rating == 75
    assert values[first_topic.topic_id].mention_count == 2
    assert values[second_topic.topic_id].rating == 25
    assert not values[second_topic.topic_id].active


def test_questionnaire_selects_unique_active_topics_deterministically() -> None:
    topics = [
        RegisteredTopic(
            topic_id=uuid4(),
            survey_id="survey",
            label=f"Тема {index}",
            keywords=[f"тема{index}"],
            centroid=[float(index), 1.0],
            rating=index + 1,
        )
        for index in range(4)
    ]

    first = build_questionnaire("survey", 2, topics, topic_question_count=2, seed=10)
    second = build_questionnaire("survey", 2, topics, topic_question_count=2, seed=10)

    assert first.source_topic_ids == second.source_topic_ids
    assert len(first.source_topic_ids) == len(set(first.source_topic_ids)) == 2
    assert all(question.topic_id for question in first.questions[-2:])


class FakeBackend:
    def fit(self, documents):
        return TopicDiscovery(
            topics=[
                DiscoveredTopic(
                    local_id=0,
                    label="Комиссия",
                    keywords=["комиссия"],
                    centroid=[1.0, 0.0],
                )
            ],
            assignments=[LocalAssignment(item.document_id, 0, 0.9) for item in documents],
            metrics={"noise_ratio": 0.0},
        )


class FakeStore:
    def __init__(self):
        interview_id = uuid4()
        self.documents = [document(interview_id, 0, "Высокая комиссия")]
        self.saved = None
        self.latest = None

    def list_topic_documents(self, survey_id):
        return self.documents

    def load_topics(self, survey_id):
        return []

    def next_questionnaire_version(self, survey_id):
        return 2 if self.latest is None else self.latest.version + 1

    def load_latest_questionnaire(self, survey_id):
        return self.latest

    def save_topic_cycle(self, result):
        self.saved = result
        if result.questionnaire_changed:
            self.latest = result.questionnaire


def test_topic_cycle_persists_topics_ratings_and_questionnaire() -> None:
    store = FakeStore()

    result = run_topic_cycle(
        store, "survey", FakeBackend(), topic_question_count=1, min_mentions=1
    )

    assert store.saved is result
    assert result.topics[0].rating == 67
    assert result.questionnaire.version == 2
    assert result.questionnaire_changed
    assert result.questionnaire.source_topic_ids == [result.topics[0].topic_id]


def test_topic_cycle_does_not_duplicate_unchanged_questionnaire() -> None:
    store = FakeStore()
    first = run_topic_cycle(
        store, "survey", FakeBackend(), topic_question_count=1, min_mentions=1
    )

    second = run_topic_cycle(
        store, "survey", FakeBackend(), topic_question_count=1, min_mentions=1
    )

    assert first.questionnaire_changed
    assert not second.questionnaire_changed
    assert second.questionnaire.version == first.questionnaire.version
