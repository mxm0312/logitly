"""A deterministic stand-in model, for tests and for examples that must not download."""

import hashlib
from collections.abc import Callable, Sequence

import numpy as np

from logitly.backends import Backend
from logitly.calibration import softmax
from logitly.core.prompt import Prompt

Oracle = Callable[[str], int]


def _seed(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")


class FakeBackend(Backend):
    """Pseudo-random but reproducible option scores.

    Args:
        oracle: Maps a rendered prompt to the index of its correct option.
            Without one the scores carry no signal, which is enough to exercise
            plumbing; with one they are informative and miscalibrated on purpose.
        strength: How much the oracle favours the correct option.
        temperature: Applied to the scores, so below 1 it produces the
            overconfidence calibration has to undo.
        bias: Added per option, the standing preference calibration removes.
    """

    name = "fake"

    def __init__(
        self,
        oracle: Oracle | None = None,
        *,
        strength: float = 2.0,
        temperature: float = 0.5,
        bias: Sequence[float] | None = None,
        vocab_size: int = 50_257,
    ):
        self.oracle = oracle
        self.strength = strength
        self.temperature = temperature
        self.bias = None if bias is None else np.asarray(bias, dtype=np.float64)
        self.vocab_size = vocab_size

    def first_token_id(self, text: str) -> int | None:
        return _seed(text) % self.vocab_size if text else None

    def score_labels(self, prompts, labels, *, progress=None) -> np.ndarray:
        return self._rows(prompts, len(labels), progress)

    def score_texts(self, prompts, options, *, length_normalize=True, progress=None) -> np.ndarray:
        return np.log(softmax(self._rows(prompts, len(options), progress)))

    def _rows(self, prompts: Sequence[Prompt], width: int, progress) -> np.ndarray:
        rows = np.stack([self._row(self.text_for(prompt), width) for prompt in prompts])
        if progress:
            progress(len(prompts), len(prompts))
        return rows

    def _row(self, text: str, width: int) -> np.ndarray:
        scores = np.random.default_rng(_seed(text)).normal(size=width)
        if self.oracle is not None:
            scores[self.oracle(text)] += self.strength
        scores /= self.temperature
        return scores if self.bias is None else scores + self.bias
