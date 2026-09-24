"""Fitting temperature and per-option bias by gradient descent.

An LLM picking between options usually ranks them well but scores them badly:
it is overconfident, and it favours some option positions whatever the input.
Both are fixed by the smallest transform that can,

    p = softmax((z - b) / T)

fitted by minimising the negative log-likelihood of a labelled set. T is
monotone per option, so only b can change which option wins.
"""

import math
from collections.abc import Sequence

import numpy as np

from logitly.calibration.transform import Calibration, softmax
from logitly.config import FitConfig, Frozen
from logitly.errors import CalibrationError


class FitResult(Frozen):
    """The fitted calibration plus the optimiser trace."""

    calibration: Calibration
    loss_history: list[float]
    iterations: int
    converged: bool


def _nll_and_grad(
    z: np.ndarray, y: np.ndarray, log_t: float, bias: np.ndarray, prior: float
) -> tuple[float, float, np.ndarray]:
    """Penalised mean NLL and its gradients wrt log T and b.

    Hand-written so calibration needs numpy only; scripts/check.py verifies it
    against finite differences.
    """
    n = len(z)
    temperature = math.exp(log_t)
    s = (z - bias) / temperature
    p = softmax(s)

    loss = float(-np.log(np.clip(p[np.arange(n), y], 1e-12, None)).mean())
    g = p
    g[np.arange(n), y] -= 1.0
    g /= n

    # The prior is divided by n so it acts like a fixed belief about one model
    # rather than a per-example penalty: decisive on 20 examples, invisible on 2000.
    weight = prior / n
    loss += 0.5 * weight * (log_t**2 + float(bias @ bias))
    grad_log_t = float(-(g * s).sum()) + weight * log_t
    grad_bias = -g.sum(axis=0) / temperature + weight * bias
    return loss, grad_log_t, grad_bias


def fit_calibration(
    scores: np.ndarray,
    labels: Sequence[int],
    *,
    options: Sequence[str] | None = None,
    model: str | None = None,
    config: FitConfig | None = None,
) -> FitResult:
    """Fit T and b by full-batch Adam on the negative log-likelihood.

    Args:
        scores: (n_examples, n_options) raw option logits.
        labels: Index of the correct option per example.
        options: Option names, recorded on the result for later validation.
        model: Model name, recorded the same way.
        config: Optimiser settings.

    Raises:
        CalibrationError: If the data is empty or malformed.
    """
    cfg = config or FitConfig()
    z = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if z.ndim != 2 or z.shape[0] == 0:
        raise CalibrationError(f"scores must be a non-empty 2-D array, got shape {z.shape}")
    if y.shape != (z.shape[0],):
        raise CalibrationError(f"expected {z.shape[0]} labels, got {y.shape}")
    if y.min() < 0 or y.max() >= z.shape[1]:
        raise CalibrationError("labels contain an index outside the option range")
    if not np.isfinite(z).all():
        raise CalibrationError("scores contain non-finite values")

    n_options = z.shape[1]
    log_t = math.log(cfg.init_temperature)
    bias = np.zeros(n_options)
    moment1 = np.zeros(n_options + 1)
    moment2 = np.zeros(n_options + 1)

    history = [_nll_and_grad(z, y, log_t, bias, cfg.prior)[0]]
    converged = False
    step = 0

    for step in range(1, cfg.max_iters + 1):
        _, grad_log_t, grad_bias = _nll_and_grad(z, y, log_t, bias, cfg.prior)
        grad = np.concatenate(([grad_log_t], grad_bias if cfg.fit_bias else np.zeros(n_options)))

        moment1 = cfg.beta1 * moment1 + (1 - cfg.beta1) * grad
        moment2 = cfg.beta2 * moment2 + (1 - cfg.beta2) * grad * grad
        corrected1 = moment1 / (1 - cfg.beta1**step)
        corrected2 = moment2 / (1 - cfg.beta2**step)
        update = cfg.learning_rate * corrected1 / (np.sqrt(corrected2) + cfg.eps)

        log_t = float(np.clip(log_t - update[0], -cfg.max_log_temperature, cfg.max_log_temperature))
        bias = bias - update[1:]
        bias -= bias.mean()  # softmax ignores a constant shift, so pin it rather than let it drift

        history.append(_nll_and_grad(z, y, log_t, bias, cfg.prior)[0])
        if abs(history[-2] - history[-1]) < cfg.tol:
            converged = True
            break

    calibration = Calibration(
        temperature=math.exp(log_t),
        bias=bias.tolist() if cfg.fit_bias else [0.0] * n_options,
        options=list(options) if options is not None else None,
        model=model,
    )
    return FitResult(
        calibration=calibration, loss_history=history, iterations=step, converged=converged
    )
