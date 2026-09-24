"""The benchmark's output: one record per measured run, and how they group."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from logitbench.config import BenchConfig
from logitbench.metrics import Bins
from logitly.config import Frozen

Method = Literal["raw", "calibrated"]


class Record(Frozen):
    """One model, one dataset, one calibration set, measured on the test set."""

    model: str
    dataset: str
    method: Method
    n_test: int
    metrics: dict[str, float]
    bins: Bins
    calib_size: int | None = None
    seed: int | None = None
    temperature: float | None = None
    bias: list[float] | None = None


class Results(Frozen):
    """A finished run: the config that produced it and every record."""

    created: str
    config: BenchConfig
    records: list[Record] = Field(min_length=1)

    @classmethod
    def of(cls, config: BenchConfig, records: list[Record]) -> "Results":
        return cls(
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            config=config,
            records=records,
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Results":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))


@dataclass(frozen=True, order=True)
class Group:
    """The records that differ only by seed, and so get averaged together."""

    model: str
    dataset: str
    calib_size: int | None

    @property
    def label(self) -> str:
        return "raw" if self.calib_size is None else f"calibrated n={self.calib_size}"


class Cell(Frozen):
    """One metric over the seeds of one group."""

    mean: float
    std: float
    n_seeds: int

    @classmethod
    def of(cls, values: list[float]) -> "Cell":
        return cls(
            mean=float(np.mean(values)),
            std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            n_seeds=len(values),
        )


def aggregate(records: list[Record]) -> dict[Group, dict[str, Cell]]:
    """Mean and standard deviation across seeds, per group and metric."""
    grouped: dict[Group, list[Record]] = defaultdict(list)
    for record in records:
        grouped[Group(record.model, record.dataset, record.calib_size)].append(record)
    return {
        group: {
            metric: Cell.of([record.metrics[metric] for record in rows])
            for metric in rows[0].metrics
        }
        for group, rows in grouped.items()
    }


def histogram(records: list[Record], group: Group) -> Bins:
    """The reliability histogram of a group, summed over its seeds."""
    rows = [
        record.bins
        for record in records
        if (record.model, record.dataset, record.calib_size)
        == (group.model, group.dataset, group.calib_size)
    ]
    return sum(rows[1:], rows[0])
