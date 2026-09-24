import numpy as np
import pytest

from logitly import CalibrationError, Choice, Example
from logitly.calibration import normalize_dataset, stratified_split


@pytest.fixture
def question():
    return Choice(instructions="q", options=["calm", "sad", "angry"])


def test_examples_and_dicts_are_both_accepted(question):
    data = [Example(input="a", answer="calm"), {"input": "b", "answer": "C"}]
    inputs, labels = normalize_dataset(question, data)
    assert inputs == ["a", "b"]
    assert labels.tolist() == [0, 2]


def test_a_mapping_input_survives_normalisation(question):
    inputs, _ = normalize_dataset(question, [{"input": {"body": "hi"}, "answer": 0}])
    assert inputs == [{"body": "hi"}]


@pytest.mark.parametrize(
    "data",
    [
        [],
        [{"text": "a", "label": "calm"}],
        [("a", "calm")],
        [{"input": "a"}],
        ["just a string"],
    ],
)
def test_unsupported_shapes_are_rejected(question, data):
    with pytest.raises(CalibrationError):
        normalize_dataset(question, data)


def test_an_unknown_answer_names_the_offending_example(question):
    with pytest.raises(CalibrationError, match="example 1"):
        normalize_dataset(
            question, [{"input": "a", "answer": "calm"}, {"input": "b", "answer": "nope"}]
        )


def test_the_split_is_disjoint_and_covers_everything():
    labels = np.array([0] * 10 + [1] * 10)
    train, test = stratified_split(labels, 0.25, seed=0)
    assert len(train) + len(test) == len(labels)
    assert not set(train) & set(test)


def test_the_split_keeps_the_label_mix():
    labels = np.array([0] * 20 + [1] * 10)
    _, test = stratified_split(labels, 0.5, seed=0)
    assert np.bincount(labels[test]).tolist() == [10, 5]


def test_a_class_with_one_example_stays_in_train():
    labels = np.array([0, 0, 0, 0, 1])
    train, test = stratified_split(labels, 0.5, seed=0)
    assert 4 in train and 4 not in test


def test_no_test_split_when_asked_for_none_or_too_little_data():
    labels = np.array([0, 1, 0, 1])
    assert len(stratified_split(labels, 0.0)[1]) == 0
    assert len(stratified_split(np.array([0, 1, 0]), 0.5)[1]) == 0


def test_the_split_is_reproducible():
    labels = np.array([0] * 10 + [1] * 10)
    assert np.array_equal(
        stratified_split(labels, 0.3, seed=7)[1], stratified_split(labels, 0.3, seed=7)[1]
    )
