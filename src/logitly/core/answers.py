"""Answers returned by ask."""

from logitly.config import Frozen


class Answer(Frozen):
    """The model's decision for one question on one input.

    Attributes:
        label: The winning option.
        index: Its position in the question's option list.
        probs: Probability per option, calibrated if the question is.
        raw_probs: Probabilities before calibration.
        scores: The raw option logits the model produced.
    """

    label: str
    index: int
    probs: dict[str, float]
    raw_probs: dict[str, float]
    scores: tuple[float, ...]
    calibrated: bool

    @property
    def options(self) -> list[str]:
        return list(self.probs)

    @property
    def confidence(self) -> float:
        """Probability of the winning option."""
        return self.probs[self.label]

    @property
    def margin(self) -> float:
        """Gap to the runner-up: a better "send this to a human" signal than confidence."""
        ordered = sorted(self.probs.values(), reverse=True)
        return ordered[0] - ordered[1]

    @property
    def ranking(self) -> list[tuple[str, float]]:
        """(option, probability) pairs, most likely first."""
        return sorted(self.probs.items(), key=lambda item: item[1], reverse=True)

    def __str__(self) -> str:
        return self.label

    def __repr__(self) -> str:
        parts = ", ".join(f"{option}={p:.3f}" for option, p in self.ranking)
        state = "calibrated" if self.calibrated else "raw"
        return f"{type(self).__name__}({self.label!r}, {state}, {parts})"


class BoolAnswer(Answer):
    """Answer to a Bool question."""

    @property
    def value(self) -> bool:
        return self.index == 0

    @property
    def p_true(self) -> float:
        return self.probs[self.options[0]]

    def __bool__(self) -> bool:
        return self.value


class ScoreAnswer(Answer):
    """Answer to a Score question."""

    @property
    def score(self) -> int:
        return self.index

    @property
    def expected_score(self) -> float:
        """Probability-weighted position on the scale, not just the argmax."""
        return sum(i * p for i, p in enumerate(self.probs.values()))
