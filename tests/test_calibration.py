import math

import numpy as np
import pytest
from pydantic import ValidationError

from logitly import Calibration, CalibrationError, FitConfig, fit_calibration
from logitly.calibration import softmax
from logitly.calibration.fit import _nll_and_grad


def miscalibrated(n=2000, k=3, temperature=2.5, bias=(0.8, -0.3, -0.5), seed=0):
    """Logits drawn so that softmax((z - b) / T) is exactly the generating model."""
    rng = np.random.default_rng(seed)
    bias = np.asarray(bias) - np.mean(bias)
    latent = rng.normal(size=(n, k)) * 2.0
    labels = np.array([rng.choice(k, p=softmax(row)) for row in latent])
    return latent * temperature + bias, labels, temperature, bias


def test_gradients_match_finite_differences():
    z, y, _, _ = miscalibrated(n=64, seed=3)
    log_t, bias, prior = 0.3, np.array([0.2, -0.1, -0.1]), 2.0
    _, grad_log_t, grad_bias = _nll_and_grad(z, y, log_t, bias, prior)

    step = 1e-6
    numeric_t = (
        _nll_and_grad(z, y, log_t + step, bias, prior)[0]
        - _nll_and_grad(z, y, log_t - step, bias, prior)[0]
    ) / (2 * step)
    assert grad_log_t == pytest.approx(numeric_t, rel=1e-5, abs=1e-8)

    for i in range(len(bias)):
        shift = np.zeros_like(bias)
        shift[i] = step
        numeric_b = (
            _nll_and_grad(z, y, log_t, bias + shift, prior)[0]
            - _nll_and_grad(z, y, log_t, bias - shift, prior)[0]
        ) / (2 * step)
        assert grad_bias[i] == pytest.approx(numeric_b, rel=1e-5, abs=1e-8)


def test_recovers_the_generating_parameters():
    z, y, temperature, bias = miscalibrated(n=4000)
    result = fit_calibration(z, y, options=list("abc"))

    assert result.converged
    assert result.calibration.temperature == pytest.approx(temperature, rel=0.1)
    assert np.allclose(result.calibration.bias, bias, atol=0.15)
    assert result.loss_history[-1] < result.loss_history[0]


def test_bias_is_zero_centred():
    z, y, _, _ = miscalibrated()
    assert sum(fit_calibration(z, y).calibration.bias) == pytest.approx(0.0, abs=1e-9)


def test_fit_bias_off_leaves_plain_temperature_scaling():
    z, y, _, _ = miscalibrated()
    result = fit_calibration(z, y, config=FitConfig(fit_bias=False))
    assert result.calibration.bias == [0.0, 0.0, 0.0]
    assert result.calibration.temperature > 1.0


def test_loss_decreases_monotonically_enough():
    z, y, _, _ = miscalibrated(n=500)
    history = fit_calibration(z, y).loss_history
    assert history[-1] < history[0]


def test_temperature_alone_cannot_change_the_argmax():
    z, _, _, _ = miscalibrated(n=50)
    calibration = Calibration(temperature=7.0, bias=[0.0, 0.0, 0.0])
    assert np.array_equal(calibration.apply(z).argmax(1), softmax(z).argmax(1))


def test_apply_matches_the_formula():
    calibration = Calibration(temperature=2.0, bias=[1.0, -1.0])
    z = np.array([[3.0, 1.0]])
    assert np.allclose(calibration.apply(z), softmax((z - np.array([1.0, -1.0])) / 2.0))


def test_identity_is_a_noop():
    identity = Calibration.identity(["a", "b"])
    assert identity.is_identity
    assert np.allclose(identity.apply(np.array([[2.0, 1.0]])), softmax(np.array([[2.0, 1.0]])))


def test_calibration_is_immutable_and_serialisable():
    calibration = Calibration(temperature=1.5, bias=[0.1, -0.1], options=["a", "b"])
    restored = Calibration.model_validate_json(calibration.model_dump_json())
    assert restored == calibration
    with pytest.raises(ValidationError):
        calibration.temperature = 2.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0.0},
        {"temperature": -1.0},
        {"temperature": 1.0, "bias": [0.1], "options": ["a", "b"]},
    ],
)
def test_invalid_calibrations_are_rejected(kwargs):
    with pytest.raises(ValidationError):
        Calibration(**kwargs)


def test_wrong_option_count_is_rejected_on_apply():
    with pytest.raises(CalibrationError):
        Calibration(temperature=1.0, bias=[0.0, 0.0]).apply(np.zeros((2, 3)))


def test_check_options_rejects_a_different_question():
    with pytest.raises(CalibrationError):
        Calibration(temperature=1.0, options=["a", "b"]).check_options(["a", "c"])


@pytest.mark.parametrize(
    ("scores", "labels"),
    [
        (np.zeros((0, 3)), []),
        (np.zeros((2, 3)), [0]),
        (np.zeros((2, 3)), [0, 9]),
        (np.full((2, 3), np.nan), [0, 1]),
    ],
)
def test_fit_rejects_malformed_data(scores, labels):
    with pytest.raises(CalibrationError):
        fit_calibration(scores, labels)


def test_the_prior_stops_a_small_separable_set_from_buying_certainty():
    """Eight examples, all correct by a thin margin: shrinking T drives NLL to zero."""
    z = np.tile([0.2, -0.2], (8, 1))
    labels = [0] * 8

    unpenalised = fit_calibration(z, labels, config=FitConfig(prior=0.0)).calibration
    penalised = fit_calibration(z, labels).calibration

    assert unpenalised.apply(z)[0].max() > 0.999, "expected the unpenalised fit to collapse"
    assert penalised.apply(z)[0].max() < 0.99
    assert penalised.temperature > unpenalised.temperature
    assert math.isfinite(penalised.temperature)


def test_the_prior_is_negligible_on_a_large_set():
    z, y, _temperature, _ = miscalibrated(n=4000)
    with_prior = fit_calibration(z, y).calibration.temperature
    without = fit_calibration(z, y, config=FitConfig(prior=0.0)).calibration.temperature
    assert with_prior == pytest.approx(without, rel=0.01)
