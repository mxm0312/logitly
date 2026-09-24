"""The fitted probability transform."""

from collections.abc import Sequence

import numpy as np
from pydantic import Field, model_validator

from logitly.config import Frozen
from logitly.errors import CalibrationError


def softmax(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    exp = np.exp(scores - np.max(scores, axis=axis, keepdims=True))
    return exp / exp.sum(axis=axis, keepdims=True)


class Calibration(Frozen):
    """A fitted softmax((z - b) / T).

    Attributes:
        temperature: > 1 softens an overconfident model, < 1 sharpens it.
        bias: One value per option, zero-centred.
        options: The options it was fitted for, so a mismatch fails loudly.
        model: The model it was fitted on. T and b describe one model's logits
            and do not transfer to another.
    """

    temperature: float = Field(default=1.0, gt=0)
    bias: list[float] | None = None
    options: list[str] | None = None
    model: str | None = None

    @model_validator(mode="after")
    def _check_widths(self) -> "Calibration":
        if (
            self.bias is not None
            and self.options is not None
            and len(self.bias) != len(self.options)
        ):
            raise ValueError(f"{len(self.bias)} biases for {len(self.options)} options")
        return self

    @classmethod
    def identity(cls, options: Sequence[str] | None = None) -> "Calibration":
        opts = list(options) if options is not None else None
        return cls(bias=[0.0] * len(opts) if opts else None, options=opts)

    @property
    def is_identity(self) -> bool:
        return self.temperature == 1.0 and not any(self.bias or ())

    def check_options(self, options: Sequence[str]) -> None:
        """Raise if this calibration belongs to a different question."""
        if self.options is not None and list(options) != self.options:
            raise CalibrationError(
                f"calibration was fitted for {self.options}, question declares {list(options)}"
            )

    def apply(self, scores: np.ndarray) -> np.ndarray:
        """Raw option scores -> calibrated probabilities."""
        z = np.asarray(scores, dtype=np.float64)
        if self.bias is not None:
            if z.shape[-1] != len(self.bias):
                raise CalibrationError(f"expected {len(self.bias)} scores, got {z.shape[-1]}")
            z = z - np.asarray(self.bias)
        return softmax(z / self.temperature)

    def __str__(self) -> str:
        if self.bias is None:
            return f"Calibration(T={self.temperature:.3f})"
        names = self.options or [str(i) for i in range(len(self.bias))]
        bias = "  ".join(f"{n}={v:+.3f}" for n, v in zip(names, self.bias, strict=True))
        return f"Calibration(T={self.temperature:.3f}, bias: {bias})"
