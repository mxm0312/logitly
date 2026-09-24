"""Where the run writes. One folder per dataset, so results stay browsable."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Layout:
    """results/
    results.json          every record, machine-readable
    report.md             settings, summary, links
    <dataset>/
        report.md         per-model tables
        plots/*.png
        scores/*.npz      cached model output
    """

    root: Path

    @property
    def results(self) -> Path:
        return self.root / "results.json"

    @property
    def report(self) -> Path:
        return self.root / "report.md"

    def dataset(self, key: str) -> Path:
        return self.root / key

    def dataset_report(self, key: str) -> Path:
        return self.dataset(key) / "report.md"

    def scores(self, key: str) -> Path:
        return self.dataset(key) / "scores"

    def plots(self, key: str) -> Path:
        return self.dataset(key) / "plots"
