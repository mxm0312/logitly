"""Calibration data: one explicit shape, validated on the way in."""

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel, TypeAdapter, ValidationError

from logitly.errors import CalibrationError, QuestionError

if TYPE_CHECKING:
    from logitly.core.questions import Question


class Example(BaseModel):
    """One labelled item.

    Attributes:
        input: What you would pass to ask -- a string, or a
            mapping of named fields.
        answer: The correct option, as its text, its label ("A") or its index.
    """

    input: Any
    answer: str | int


DATASET_ADAPTER: TypeAdapter[list[Example]] = TypeAdapter(list[Example])


def normalize_dataset(question: "Question", data: Iterable[Any]) -> tuple[list[Any], np.ndarray]:
    """Validate a calibration set and resolve its answers to option indices.

    Args:
        data: Example objects, or dicts with input and answer keys.

    Raises:
        CalibrationError: If the data is empty, misshapen, or an answer is not
            one of the question's options.
    """
    try:
        examples = DATASET_ADAPTER.validate_python(list(data))
    except ValidationError as exc:
        raise CalibrationError(f"invalid calibration dataset:\n{exc}") from exc
    if not examples:
        raise CalibrationError("the calibration dataset is empty")

    labels = []
    for position, example in enumerate(examples):
        try:
            labels.append(question.index_of(example.answer))
        except QuestionError as exc:
            raise CalibrationError(f"example {position}: {exc}") from exc
    return [example.input for example in examples], np.asarray(labels, dtype=np.int64)


def stratified_split(
    labels: Sequence[int], test_size: float, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle and split indices, keeping each side's label mix intact.

    A class with a single example stays in train: held out it would teach you
    nothing and cost you a training point.
    """
    y = np.asarray(labels, dtype=np.int64)
    n = len(y)
    if test_size == 0.0 or n < 4:
        return np.arange(n), np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(seed)
    test: list[int] = []
    for value in np.unique(y):
        idx = np.flatnonzero(y == value)
        rng.shuffle(idx)
        if len(idx) >= 2:
            n_test = max(1, min(round(test_size * len(idx)), len(idx) - 1))
            test.extend(idx[:n_test].tolist())

    test_idx = np.sort(np.asarray(test, dtype=np.int64))
    return np.setdiff1d(np.arange(n), test_idx), test_idx
