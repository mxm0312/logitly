"""Question types: what you ask, and which answers you will accept."""

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from logitly.calibration import Calibration, softmax
from logitly.config import PromptConfig, Scoring
from logitly.core.answers import Answer, BoolAnswer, ScoreAnswer
from logitly.core.prompt import LETTERS, option_labels
from logitly.errors import QuestionError


class Question(BaseModel):
    """A closed question: fixed instructions, a fixed menu of answers.

    The menu is the point. Because the answer space is known up front we read
    probabilities straight off the logits instead of parsing generated text, and
    those probabilities can then be calibrated.

    Attributes:
        instructions: What the model is asked.
        options: The allowed answers. Order matters: the calibration bias is
            indexed by it.
        context: Background constant across calls (the system prompt in chat mode).
        scoring: "letter" reads the logits of the option labels in one forward
            pass; "text" scores the full option strings, costing one pass per
            option but not depending on the model following a label format.
        prompt: Per-question formatting override.
        calibration: Attached by calibrate.
        name: Key used by ask_all.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    instructions: str = Field(min_length=1)
    options: list[str] = Field(min_length=2)
    context: str | None = None
    scoring: Scoring = "letter"
    prompt: PromptConfig | None = None
    calibration: Calibration | None = None
    name: str | None = None

    answer_class: ClassVar[type[Answer]] = Answer

    @field_validator("options")
    @classmethod
    def _check_options(cls, options: list[str]) -> list[str]:
        options = [option.strip() for option in options]
        if any(not option for option in options):
            raise ValueError("options must not be blank")
        duplicates = sorted({o for o in options if options.count(o) > 1})
        if duplicates:
            raise ValueError(f"options must be unique, repeated: {duplicates}")
        return options

    @model_validator(mode="after")
    def _check_calibration(self) -> "Question":
        if self.calibration is not None:
            self.calibration.check_options(self.options)
        return self

    @property
    def n_options(self) -> int:
        return len(self.options)

    @property
    def is_calibrated(self) -> bool:
        return self.calibration is not None and not self.calibration.is_identity

    def labels(self, label_style: str = "letter") -> list[str]:
        """The short labels shown next to the options."""
        return option_labels(len(self.options), label_style)

    def index_of(self, value: str | int) -> int:
        """Resolve a gold answer to an option index.

        Accepts the option text (case-insensitive), its label ("A", "1"),
        or an integer index.
        """
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            if not 0 <= int(value) < len(self.options):
                raise QuestionError(f"option index {value} out of range for {self.options}")
            return int(value)

        text = str(value).strip().lower()
        groups = [self.options, self.labels("number")]
        if len(self.options) <= len(LETTERS):
            groups.insert(1, self.labels("letter"))
        for group in groups:
            for index, candidate in enumerate(group):
                if text == candidate.lower():
                    return index
        raise QuestionError(f"{value!r} is not one of {self.options}")

    def make_answer(self, scores: Sequence[float]) -> Answer:
        """Build an Answer from raw option scores."""
        raw = np.asarray(scores, dtype=np.float64).reshape(-1)
        if raw.shape[0] != len(self.options):
            raise QuestionError(f"expected {len(self.options)} scores, got {raw.shape[0]}")
        raw_probs = softmax(raw)
        probs = self.calibration.apply(raw) if self.calibration else raw_probs
        index = int(np.argmax(probs))
        return self.answer_class(
            label=self.options[index],
            index=index,
            probs=dict(zip(self.options, probs.tolist(), strict=True)),
            raw_probs=dict(zip(self.options, raw_probs.tolist(), strict=True)),
            scores=tuple(raw.tolist()),
            calibrated=self.calibration is not None,
        )

    def save(self, path: str | Path) -> None:
        """Write the question and its calibration to JSON."""
        Path(path).write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Question":
        """Read back a question saved with save."""
        return QUESTION_ADAPTER.validate_json(Path(path).read_text(encoding="utf-8"))

    def __str__(self) -> str:
        state = "calibrated" if self.is_calibrated else "uncalibrated"
        return f"{type(self).__name__}({self.instructions!r}, {self.options}, {state})"


class Choice(Question):
    """Pick exactly one of several named options."""

    kind: Literal["choice"] = "choice"


class Bool(Question):
    """A yes/no question; the answer exposes .value."""

    kind: Literal["bool"] = "bool"
    options: list[str] = ["yes", "no"]
    answer_class: ClassVar[type[Answer]] = BoolAnswer

    @field_validator("options")
    @classmethod
    def _check_pair(cls, options: list[str]) -> list[str]:
        if len(options) != 2:
            raise ValueError(f"Bool needs exactly 2 options, got {options}")
        return options


class Score(Question):
    """Pick a level on an *ordered* scale, lowest first.

    The answer also exposes .expected_score, the probability-weighted
    position on the scale.
    """

    kind: Literal["score"] = "score"
    answer_class: ClassVar[type[Answer]] = ScoreAnswer

    @property
    def levels(self) -> list[str]:
        return self.options


AnyQuestion = Annotated[Choice | Bool | Score, Field(discriminator="kind")]
QUESTION_ADAPTER: TypeAdapter[Any] = TypeAdapter(AnyQuestion)
