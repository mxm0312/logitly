"""Figures, one folder per dataset. Nothing here reads anything but the records."""

from math import ceil, sqrt
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from logitbench.layout import Layout
from logitbench.metrics import LABELS
from logitbench.records import Group, Results, aggregate, histogram

matplotlib.use("Agg")  # figures are written to disk, never shown


def write_plots(results: Results, layout: Layout) -> None:
    """A curve per metric and a reliability diagram per model, then the summary figure."""
    cells = aggregate(results.records)
    sizes = sorted(results.config.experiment.calib_sizes)
    largest = max(sizes)

    for dataset in {group.dataset for group in cells}:
        directory = layout.plots(dataset)
        directory.mkdir(parents=True, exist_ok=True)
        models = [key for key in results.config.models if Group(key, dataset, None) in cells]

        for metric in results.config.report.curve_metrics:
            figure = _curve(cells, dataset, models, sizes, metric)
            _save(figure, directory / f"{metric}.png", results.config.report.dpi)
        if results.config.report.reliability:
            for model in models:
                figure = _reliability(results, cells, dataset, model, largest)
                _save(figure, directory / f"reliability_{model}.png", results.config.report.dpi)
    write_summary(results, layout)


def _curve(cells, dataset: str, models: list[str], sizes: list[int], metric: str):
    """Metric against calibration-set size; the dashed line is the same model uncalibrated."""
    figure, axes = plt.subplots(figsize=(6.4, 4.0))
    for model, colour in zip(models, _palette(len(models)), strict=True):
        points = [cells[Group(model, dataset, size)][metric] for size in sizes]
        mean = np.array([point.mean for point in points])
        std = np.array([point.std for point in points])
        axes.plot(sizes, mean, marker="o", color=colour, label=model)
        axes.fill_between(sizes, mean - std, mean + std, color=colour, alpha=0.15, linewidth=0)
        axes.axhline(
            cells[Group(model, dataset, None)][metric].mean, color=colour, ls="--", lw=1, alpha=0.7
        )

    axes.set(
        xscale="log",
        xlabel="calibration examples",
        ylabel=_label(metric),
        title=f"{dataset} — {_label(metric)}",
    )
    axes.set_xticks(sizes, [str(size) for size in sizes])
    axes.grid(alpha=0.25, linewidth=0.6)
    axes.legend(title="dashed = uncalibrated", fontsize=8, title_fontsize=8)
    return figure


def _reliability(results: Results, cells, dataset: str, model: str, largest: int):
    """Confidence against accuracy, raw and calibrated, pooled over seeds."""
    figure, (top, bottom) = plt.subplots(
        2, 1, figsize=(5.2, 5.4), sharex=True, height_ratios=[3, 1], constrained_layout=True
    )
    top.plot([0, 1], [0, 1], color="0.6", ls=":", lw=1)
    for group, colour in (
        (Group(model, dataset, None), "tab:red"),
        (Group(model, dataset, largest), "tab:blue"),
    ):
        confidence, accuracy, weight = histogram(results.records, group).means()
        label = f"{group.label} (ECE {cells[group]['ece'].mean:.3f})"
        top.plot(confidence, accuracy, marker="o", ms=4, color=colour, label=label)
        bottom.plot(confidence, weight, marker="o", ms=3, color=colour)

    top.set(ylabel="accuracy", title=f"{dataset} — {model}", xlim=(0, 1), ylim=(0, 1))
    top.legend(fontsize=8, loc="upper left")
    top.grid(alpha=0.25, linewidth=0.6)
    bottom.set(xlabel="confidence", ylabel="share")
    bottom.grid(alpha=0.25, linewidth=0.6)
    return figure


def _palette(n: int) -> list:
    return [plt.get_cmap("tab10")(i % 10) for i in range(n)]


def _label(metric: str) -> str:
    return LABELS[metric]


def _save(figure, path, dpi: int) -> None:
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def write_summary(results: Results, layout: Layout) -> Path:
    """One figure for the README: what calibration fixes, and how much data it needs."""
    cells = aggregate(results.records)
    sizes = sorted(results.config.experiment.calib_sizes)
    present = {(group.model, group.dataset) for group in cells}
    models = [key for key in results.config.models if any(key == m for m, _ in present)]
    datasets = [key for key in results.config.datasets if any(key == d for _, d in present)]

    columns = ceil(sqrt(len(datasets)))
    rows = ceil(len(datasets) / columns)
    figure = plt.figure(figsize=(6.2 + 2.9 * columns, max(4.4, 2.4 * rows)), layout="constrained")
    grid = figure.add_gridspec(rows, columns + 2)

    worst = max(present, key=lambda pair: cells[Group(*pair, None)]["ece"].mean)
    _reliability_panel(
        figure.add_subplot(grid[:, :2]),
        results,
        cells,
        Group(*worst, None),
        Group(*worst, max(sizes)),
    )
    for index, dataset in enumerate(datasets):
        axes = figure.add_subplot(grid[index // columns, 2 + index % columns])
        _size_panel(axes, cells, dataset, models, sizes, legend=index == 0 and len(models) > 1)

    path = layout.root / "summary.png"
    _save(figure, path, results.config.report.dpi)
    return path


def _reliability_panel(axes, results: Results, cells, raw: Group, fitted: Group) -> None:
    """Confidence against accuracy: the gap to the diagonal is the miscalibration."""
    axes.plot([0, 1], [0, 1], color="0.55", ls=":", lw=1.2, zorder=1)
    for group, colour, name in (
        (raw, "tab:red", "uncalibrated"),
        (fitted, "tab:blue", f"calibrated on {fitted.calib_size}"),
    ):
        confidence, accuracy, _ = histogram(results.records, group).means()
        axes.plot(
            confidence,
            accuracy,
            marker="o",
            ms=5,
            color=colour,
            zorder=3,
            label=f"{name} — ECE {cells[group]['ece'].mean:.3f}",
        )
        if group is raw:
            axes.fill_between(confidence, accuracy, confidence, color=colour, alpha=0.12, zorder=2)

    axes.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="stated confidence",
        ylabel="actual accuracy",
        title=f"{raw.dataset} — {raw.model}",
    )
    axes.legend(loc="upper left", fontsize=9)
    axes.grid(alpha=0.25, linewidth=0.6)


def _size_panel(
    axes, cells, dataset: str, models: list[str], sizes: list[int], legend: bool
) -> None:
    """One dataset on its own scale: ECE against the size of the calibration set."""
    single = len(models) == 1
    for model, colour in zip(models, _palette(len(models)), strict=True):
        line = "#1f3b73" if single else colour
        points = [cells[Group(model, dataset, size)]["ece"] for size in sizes]
        mean = np.array([point.mean for point in points])
        std = np.array([point.std for point in points])
        raw = cells[Group(model, dataset, None)]["ece"].mean

        axes.axhline(raw, color=line, ls=(0, (4, 3)), lw=1.1, zorder=2)
        axes.fill_between(sizes, mean - std, mean + std, color=line, alpha=0.12, lw=0, zorder=1)
        if single:  # green below the uncalibrated line, red above it: better and worse at a glance
            for colour_fill, better in (("tab:green", True), ("tab:red", False)):
                axes.fill_between(
                    sizes,
                    mean,
                    raw,
                    where=(mean < raw) == better,
                    interpolate=True,
                    color=colour_fill,
                    alpha=0.16,
                    lw=0,
                    zorder=1,
                )
        axes.plot(sizes, mean, marker="o", ms=4, lw=1.8, color=line, zorder=3, label=model)

    axes.set_title(dataset, fontsize=10)
    axes.set_xscale("log")
    axes.set_xticks(sizes, [str(size) for size in sizes], fontsize=7.5)
    axes.tick_params(axis="y", labelsize=7.5)
    axes.margins(y=0.18)
    axes.set_ylim(bottom=0)

    # Label the uncalibrated line inside the panel: below it when it sits near the top.
    high = raw > 0.75 * axes.get_ylim()[1]
    axes.annotate(
        f"uncalibrated {raw:.3f}",
        (sizes[0], raw),
        textcoords="offset points",
        xytext=(1, -10 if high else 4),
        fontsize=7.5,
        color="0.35",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1},
    )
    axes.set_xlabel("calibration examples", fontsize=8)
    axes.set_ylabel("ECE", fontsize=8)
    axes.grid(alpha=0.22, linewidth=0.6)
    if legend:
        axes.legend(fontsize=7.5)
