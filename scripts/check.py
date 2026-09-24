"""Self-check for logitly: numerical invariants offline, then a real model.

    python scripts/check.py                 # maths and plumbing, no model needed
    python scripts/check.py --model gpt2    # also runs a real Hugging Face model

Exits non-zero on the first failure.
"""

import argparse
import sys
from collections.abc import Callable

import numpy as np
import torch

from logitly import CalibrationConfig, Choice, Logitly
from logitly.calibration import softmax
from logitly.calibration.fit import _nll_and_grad, fit_calibration
from logitly.testing import FakeBackend

TONE = ["calm", "frustrated", "angry"]
CHECKS: list[tuple[str, Callable[[], str]]] = []


def check(name: str) -> Callable[[Callable[[], str]], Callable[[], str]]:
    def register(fn: Callable[[], str]) -> Callable[[], str]:
        CHECKS.append((name, fn))
        return fn

    return register


def synthetic(n=4000, k=3, temperature=2.5, bias=(0.8, -0.3, -0.5), seed=0):
    rng = np.random.default_rng(seed)
    bias = np.asarray(bias) - np.mean(bias)
    latent = rng.normal(size=(n, k)) * 2.0
    labels = np.array([rng.choice(k, p=softmax(row)) for row in latent])
    return latent * temperature + bias, labels, temperature, bias


@check("analytic gradients match finite differences")
def _gradients() -> str:
    z, y, _, _ = synthetic(n=64, seed=3)
    log_t, bias, prior = 0.3, np.array([0.2, -0.1, -0.1]), 2.0
    _, grad_log_t, grad_bias = _nll_and_grad(z, y, log_t, bias, prior)
    step = 1e-6

    def loss(t, b):
        return _nll_and_grad(z, y, t, b, prior)[0]

    numeric = [(loss(log_t + step, bias) - loss(log_t - step, bias)) / (2 * step)]
    for i in range(len(bias)):
        shift = np.zeros_like(bias)
        shift[i] = step
        numeric.append((loss(log_t, bias + shift) - loss(log_t, bias - shift)) / (2 * step))

    error = np.abs(np.array([grad_log_t, *grad_bias]) - numeric).max()
    assert error < 1e-6, f"gradient error {error:.2e} is too large"
    return f"max error {error:.2e}"


@check("gradient descent recovers the generating T and b")
def _recovery() -> str:
    z, y, temperature, bias = synthetic()
    fitted = fit_calibration(z, y, options=TONE).calibration
    assert abs(fitted.temperature - temperature) < 0.25, fitted
    assert np.allclose(fitted.bias, bias, atol=0.15), fitted
    error = np.abs(np.array(fitted.bias) - bias).max()
    return f"T {fitted.temperature:.2f} vs {temperature}, bias error {error:.3f}"


@check("calibration lowers held-out nll, ece and brier")
def _end_to_end() -> str:
    options = TONE
    rng = np.random.default_rng(0)
    picks = rng.integers(0, len(options), size=300)
    data = [
        {"input": f"ticket {i} is {options[p]}", "answer": options[p]} for i, p in enumerate(picks)
    ]

    def oracle(prompt: str) -> int:
        return next(i for i, option in enumerate(options) if f"is {option}" in prompt)

    llm = Logitly(FakeBackend(oracle=oracle, temperature=0.35, bias=[1.5, 0.0, -1.5]))
    question = Choice(instructions="What is the tone?", options=options)
    report = llm.calibrate(question, data, verbose=False)

    for metric in ("nll", "ece", "brier"):
        before, after = getattr(report.before, metric), getattr(report.after, metric)
        assert after < before, f"{metric} got worse: {before:.4f} -> {after:.4f}"
    return (
        f"nll {report.before.nll:.3f}->{report.after.nll:.3f}, "
        f"ece {report.before.ece:.3f}->{report.after.ece:.3f}"
    )


@check("probabilities are a distribution over the declared options")
def _distribution() -> str:
    llm = Logitly(FakeBackend())
    question = Choice(instructions="What is the tone?", options=TONE)
    answer = llm.ask(question, "I was charged twice")
    assert list(answer.probs) == TONE
    assert abs(sum(answer.probs.values()) - 1.0) < 1e-12
    assert answer.label == max(answer.probs, key=answer.probs.get)
    return f"{answer.label} at {answer.confidence:.0%}"


def run_model_check(model_name: str) -> int:
    """Ask a real Hugging Face model a real question. Returns the failure count."""
    print(f"\nreal model: {model_name}")
    llm = Logitly(model_name, batch_size=4)
    question = Choice(
        instructions="Is this review positive or negative?",
        options=["positive", "negative"],
        context="You label product reviews.",
    )
    reviews = [review for review, _ in _REVIEWS]

    print("\nprompt sent to the model:")
    print("-" * 58)
    print(llm.render(question, reviews[0]))
    print("-" * 58)

    failures = 0
    batched = llm.scores(question, reviews)
    one_at_a_time = np.concatenate([llm.scores(question, [review]) for review in reviews])
    drift = np.abs(batched - one_at_a_time).max()
    # Reduced precision has its own noise floor; fp32 should be near exact.
    tolerance = 1e-3 if llm.backend.model.dtype == torch.float32 else 0.25
    if drift < tolerance:
        print(f"  [ok]   left padding leaves batched scores unchanged (max drift {drift:.1e})")
    else:
        failures += 1
        print(f"  [FAIL] batching changes the scores by {drift:.3e}; check padding side")

    text_scored = Choice(
        instructions=question.instructions, options=question.options, scoring="text"
    )
    probs = llm.ask(text_scored, reviews[0]).probs
    if abs(sum(probs.values()) - 1.0) < 1e-9:
        print(f"  [ok]   text scoring returns a distribution ({probs})")
    else:
        failures += 1
        print(f"  [FAIL] text scoring returned {probs}")

    print("\nbefore calibration:")
    _show(llm, question, reviews[:2])
    llm.calibrate(
        question,
        [{"input": r, "answer": a} for r, a in _REVIEWS],
        config=CalibrationConfig(test_size=0.4),
    )
    print("after calibration:")
    _show(llm, question, reviews[:2])
    return failures


def _show(llm: Logitly, question: Choice, reviews: list[str]) -> None:
    for review, answer in zip(reviews, llm.ask_many(question, reviews), strict=True):
        print(f"  {review[:42]:<44} -> {answer.label:<9} {answer.confidence:.1%}")


_REVIEWS = [
    ("Works exactly as described, very happy.", "positive"),
    ("Arrived early and the quality is great.", "positive"),
    ("Best purchase I have made this year.", "positive"),
    ("Sturdy, cheap and does the job.", "positive"),
    ("Sound quality is excellent for the price.", "positive"),
    ("My kids love it, we use it daily.", "positive"),
    ("Broke on the first day, total waste.", "negative"),
    ("Cheap plastic, stopped working immediately.", "negative"),
    ("Never arrived and support ignored me.", "negative"),
    ("Terrible battery, dies within an hour.", "negative"),
    ("Does not fit, the photos are misleading.", "negative"),
    ("Returned it, far too noisy to use.", "negative"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Hugging Face model id to run a real check against")
    args = parser.parse_args()

    print("logitly self-check")
    failures = 0
    for name, fn in CHECKS:
        try:
            print(f"  [ok]   {name} ({fn()})")
        except AssertionError as exc:
            failures += 1
            print(f"  [FAIL] {name}: {exc}")

    if args.model and not failures:
        failures += run_model_check(args.model)

    print("\nall checks passed" if not failures else f"\n{failures} check(s) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
