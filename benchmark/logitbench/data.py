"""Datasets in, one Task out: a question, a test set, and a calibration pool."""

from dataclasses import dataclass

import numpy as np
from datasets import ClassLabel, load_dataset

from logitbench.config import DatasetSpec, ExperimentSpec
from logitbench.sampling import sample
from logitly import Choice


@dataclass(frozen=True)
class Split:
    inputs: list[str]
    labels: np.ndarray

    def take(self, rows: np.ndarray) -> "Split":
        return Split([self.inputs[i] for i in rows], self.labels[rows])


@dataclass(frozen=True)
class Task:
    """Everything one (dataset) column of the benchmark needs."""

    key: str
    question: Choice
    test: Split
    pool: Split


def load_task(key: str, spec: DatasetSpec, experiment: ExperimentSpec) -> Task:
    """Draw the test set and the calibration pool. The two never overlap."""
    rng = np.random.default_rng(experiment.data_seed)
    test_rows = _read(spec, spec.test_split)
    test_idx = sample(test_rows.labels, spec.test_size or experiment.test_size, rng)

    # The test set keeps the dataset's own class mix; the pool follows whatever the
    # calibration sets will be drawn with, so a balanced draw cannot run out of a class.
    same_split = spec.calib_split == spec.test_split
    pool_rows = test_rows if same_split else _read(spec, spec.calib_split)
    free = np.setdiff1d(np.arange(len(pool_rows.labels)), test_idx) if same_split else None
    pool_idx = sample(
        pool_rows.labels,
        experiment.calib_pool,
        rng,
        balanced=experiment.balanced_calibration,
        where=free,
    )

    return Task(key, spec.question(key), test_rows.take(test_idx), pool_rows.take(pool_idx))


def _read(spec: DatasetSpec, split: str) -> Split:
    """Load a split and check that its label ids line up with the declared options."""
    data = load_dataset(spec.path, spec.config, split=split, **spec.load_kwargs)
    feature = data.features[spec.label_field]
    if isinstance(feature, ClassLabel) and len(feature.names) != len(spec.options):
        raise ValueError(
            f"{spec.path}:{split} declares {len(feature.names)} classes {feature.names}, "
            f"but the config lists {len(spec.options)} options"
        )

    labels = np.asarray(data[spec.label_field], dtype=np.int64)
    if labels.min() < 0 or labels.max() >= len(spec.options):
        raise ValueError(
            f"{spec.path}:{split} has label ids in [{labels.min()}, {labels.max()}], "
            f"outside the {len(spec.options)} declared options"
        )
    return Split([str(text) for text in data[spec.text_field]], labels)
