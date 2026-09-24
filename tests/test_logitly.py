import numpy as np
import pytest

from helpers import OPTIONS, make_dataset, oracle_from
from logitly import BackendError, Bool, CalibrationConfig, CalibrationError, Choice, Logitly
from logitly.testing import FakeBackend


def test_ask_returns_a_full_distribution(llm, question):
    answer = llm.ask(question, "a ticket")
    assert answer.label in OPTIONS
    assert sum(answer.probs.values()) == pytest.approx(1.0)
    assert answer.scores and not answer.calibrated


def test_the_same_input_always_gets_the_same_answer(llm, question):
    assert llm.ask(question, "a ticket") == llm.ask(question, "a ticket")


def test_ask_many_matches_ask_one_by_one(llm, question):
    states = ["one", "two", "three"]
    assert llm.ask_many(question, states) == [llm.ask(question, s) for s in states]


def test_ask_many_on_nothing(llm, question):
    assert llm.ask_many(question, []) == []


def test_ask_all_keys_by_name_or_mapping_key(llm, question):
    billing = Bool(instructions="is it billing?", name="billing")
    assert set(llm.ask_all([question, billing], "x")) == {"tone", "billing"}
    assert set(llm.ask_all({"a": question, "b": billing}, "x")) == {"a", "b"}


def test_text_scoring_also_produces_a_distribution(llm):
    question = Choice(instructions="q", options=["calm", "angry"], scoring="text")
    answer = llm.ask(question, "a ticket")
    assert sum(answer.probs.values()) == pytest.approx(1.0)


def test_scoring_together_matches_scoring_one_by_one(llm, question):
    """How many inputs travel together must not change any of their scores."""
    states = [f"ticket {i}" for i in range(7)]
    together = llm.scores(question, states)
    separately = np.concatenate([llm.scores(question, [state]) for state in states])
    assert np.allclose(together, separately)


def test_a_backend_takes_no_loading_arguments():
    with pytest.raises(BackendError):
        Logitly(FakeBackend(), device="cuda")


def calibrating_llm():
    """A model that is informative, overconfident and biased towards one option."""
    backend = FakeBackend(oracle=oracle_from(), temperature=0.35, bias=[1.5, 0.0, -1.5])
    return Logitly(backend)


def test_calibration_improves_the_held_out_likelihood(question, capsys):
    report = calibrating_llm().calibrate(question, make_dataset(200))
    assert report.n_test > 0 and report.held_out
    assert report.improved
    assert report.after.ece < report.before.ece
    assert "Calibration report" in capsys.readouterr().out


def test_calibration_undoes_the_planted_overconfidence_and_preference(question):
    report = calibrating_llm().calibrate(question, make_dataset(400), verbose=False)
    calibration = report.fit.calibration
    assert calibration.temperature > 1.0
    assert np.argmax(calibration.bias) == 0 and np.argmin(calibration.bias) == 2


def test_a_model_sharpened_fourfold_needs_a_fourfold_temperature(question):
    """Scaling every logit by c scales the optimal temperature by exactly c."""
    data = make_dataset(300)
    fits = [
        Logitly(FakeBackend(oracle=oracle_from(), temperature=t)).calibrate(
            question, data, verbose=False, attach=False
        )
        for t in (1.0, 0.25)
    ]
    soft, sharp = (report.fit.calibration.temperature for report in fits)
    assert sharp == pytest.approx(4 * soft, rel=0.05)


def test_calibration_attaches_to_the_question_and_changes_later_answers(question):
    llm = calibrating_llm()
    before = llm.ask(question, "ticket 1 sounds calm")
    llm.calibrate(question, make_dataset(120), verbose=False)

    assert question.is_calibrated
    after = llm.ask(question, "ticket 1 sounds calm")
    assert after.calibrated
    assert after.confidence < before.confidence
    assert after.raw_probs == pytest.approx(before.probs)


def test_attach_false_leaves_the_question_alone(question):
    calibrating_llm().calibrate(question, make_dataset(60), verbose=False, attach=False)
    assert question.calibration is None


def test_calibrating_from_precomputed_scores_matches_the_full_run(question):
    llm = calibrating_llm()
    data = make_dataset(80)
    from_data = llm.calibrate(question, data, verbose=False, attach=False)

    inputs = [item["input"] for item in data]
    scores = llm.scores(question, inputs)
    from_scores = llm.calibrate_from_scores(
        question, scores, [item["answer"] for item in data], verbose=False, attach=False
    )
    assert from_scores.fit.calibration == from_data.fit.calibration


def test_no_test_split_is_reported_as_such(question):
    report = calibrating_llm().calibrate(
        question, make_dataset(40), config=CalibrationConfig(test_size=0.0), verbose=False
    )
    assert report.n_test == 0 and not report.held_out
    assert "TRAIN split" in str(report)


def test_mismatched_scores_and_labels_are_rejected(question):
    with pytest.raises(CalibrationError):
        calibrating_llm().calibrate_from_scores(question, np.zeros((3, 3)), [0, 1])


def test_a_calibration_fitted_on_another_model_warns(question):
    calibrating_llm().calibrate(question, make_dataset(60), verbose=False)
    elsewhere = FakeBackend()
    elsewhere.name = "some-other-model"

    with pytest.warns(UserWarning, match="fitted on"):
        Logitly(elsewhere).ask(question, "a ticket")  # no oracle, so any input scores


def test_the_model_it_was_fitted_on_is_recorded(question, recwarn):
    llm = calibrating_llm()
    llm.calibrate(question, make_dataset(60), verbose=False)

    assert question.calibration.model == llm.name
    llm.ask(question, "ticket 1 sounds calm")
    assert not recwarn.list


def test_the_report_reads_back_as_json(question):
    report = calibrating_llm().calibrate(question, make_dataset(60), verbose=False)
    assert report.model_dump()["fit"]["calibration"]["temperature"] > 0
    assert "temperature" in str(report)
