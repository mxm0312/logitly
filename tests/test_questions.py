import numpy as np
import pytest
from pydantic import ValidationError

from logitly import Bool, Calibration, CalibrationError, Choice, Question, QuestionError, Score


def test_scores_become_probabilities_in_option_order():
    question = Choice(instructions="q", options=["a", "b", "c"])
    answer = question.make_answer([2.0, 1.0, 0.0])
    assert answer.label == "a"
    assert answer.index == 0
    assert list(answer.probs) == ["a", "b", "c"]
    assert sum(answer.probs.values()) == pytest.approx(1.0)
    assert answer.confidence > answer.margin > 0


def test_calibration_changes_the_answer_but_not_the_raw_probabilities():
    question = Choice(instructions="q", options=["a", "b"])
    raw = question.make_answer([0.2, 0.0])
    question.calibration = Calibration(temperature=1.0, bias=[5.0, -5.0], options=["a", "b"])
    calibrated = question.make_answer([0.2, 0.0])

    assert raw.label == "a" and not raw.calibrated
    assert calibrated.label == "b" and calibrated.calibrated
    assert calibrated.raw_probs == pytest.approx(raw.probs)


def test_bool_exposes_a_truth_value():
    answer = Bool(instructions="is it billing?").make_answer([1.0, 0.0])
    assert answer.value is True
    assert bool(answer) is True
    assert answer.p_true == answer.confidence


def test_score_exposes_the_expected_level():
    question = Score(instructions="urgency", options=["low", "mid", "high"])
    answer = question.make_answer([0.0, 0.0, 0.0])
    assert question.levels == ["low", "mid", "high"]
    assert answer.expected_score == pytest.approx(1.0)


def test_answers_to_the_same_question_carry_the_option_order():
    question = Choice(instructions="q", options=["b", "a"])
    assert question.make_answer([1.0, 0.0]).options == ["b", "a"]


@pytest.mark.parametrize(
    "value, expected", [("angry", 2), ("ANGRY", 2), ("C", 2), ("c", 2), ("3", 2), (2, 2), (0, 0)]
)
def test_gold_answers_are_accepted_in_several_notations(value, expected):
    question = Choice(instructions="q", options=["calm", "sad", "angry"])
    assert question.index_of(value) == expected


@pytest.mark.parametrize("value", ["furious", "Z", 3, -1])
def test_unknown_gold_answers_are_rejected(value):
    question = Choice(instructions="q", options=["calm", "sad", "angry"])
    with pytest.raises(QuestionError):
        question.index_of(value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"instructions": "", "options": ["a", "b"]},
        {"instructions": "q", "options": ["a"]},
        {"instructions": "q", "options": ["a", "a"]},
        {"instructions": "q", "options": ["a", " "]},
        {"instructions": "q", "options": ["a", "b"], "scoring": "vibes"},
    ],
)
def test_invalid_questions_are_rejected(kwargs):
    with pytest.raises(ValidationError):
        Choice(**kwargs)


def test_bool_needs_exactly_two_options():
    with pytest.raises(ValidationError):
        Bool(instructions="q", options=["yes", "no", "maybe"])


def test_a_calibration_for_other_options_is_rejected():
    with pytest.raises(CalibrationError):
        Choice(
            instructions="q",
            options=["a", "b"],
            calibration=Calibration(temperature=1.0, options=["x", "y"]),
        )


def test_roundtrip_keeps_the_subclass_and_the_calibration(tmp_path):
    question = Score(
        instructions="urgency",
        options=["low", "high"],
        calibration=Calibration(temperature=1.7, bias=[0.2, -0.2], options=["low", "high"]),
        name="urgency",
    )
    path = tmp_path / "q.json"
    question.save(path)
    restored = Question.load(path)

    assert isinstance(restored, Score)
    assert restored == question
    assert restored.is_calibrated
    assert np.allclose(
        restored.make_answer([1.0, 0.0]).probs["low"], question.make_answer([1.0, 0.0]).probs["low"]
    )


def test_attaching_a_mismatched_calibration_later_is_rejected():
    question = Choice(instructions="q", options=["a", "b"])
    with pytest.raises(CalibrationError):
        question.calibration = Calibration(temperature=1.0, options=["x", "y"])


def test_wrong_number_of_scores_is_rejected():
    with pytest.raises(QuestionError):
        Choice(instructions="q", options=["a", "b"]).make_answer([1.0, 2.0, 3.0])
