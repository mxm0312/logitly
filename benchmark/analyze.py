# Figure labels use typographic × and − on purpose.
# ruff: noqa: RUF001, RUF002, RUF003

"""Cross-model figures and tables for the READMEs, built from results.json alone.

    uv run --group bench analyze.py                  # results/ from config.yml
    uv run --group bench analyze.py path/to/results.json

Writes results/analysis/*.png and results/analysis/tables.md. No model is loaded.

Reliability bins are stored as sums, so they add up across seeds and datasets:
the pooled figures weight every test example equally, and bins holding a sliver
of the examples are drawn faint rather than trusted.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from logitbench import BenchConfig, Layout, Results
from logitbench.metrics import Bins
from logitbench.records import Record

matplotlib.use("Agg")

DEFAULT_CONFIG = Path(__file__).with_name("config.yml")

# Presentation only: a config key without an entry here is shown as-is.
DISPLAY = {
    "qwen3.5-2b": "Qwen3.5-2B",
    "qwen3-vl-8b-vllm": "Qwen3-VL-8B",
    "qwen3.5-122b-vllm": "Qwen3.5-122B-A10B",
}
MODEL_COLOURS = ["#E08E0B", "#1A9E77", "#6A4C93", "#C0392B", "#2C7FB8"]
RAW, CAL = "#D1495B", "#1F6FB2"
INK, MUTED, GRID = "#1F2328", "#6E7781", "#E6E8EB"
FAINT_SHARE = 0.005  # bins with less than 0.5% of the examples are noise
THRESHOLDS = (0.8, 14 / 15)  # confidence cut-offs quoted in the tables: exact bin edges

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.edgecolor": "#8C959F",
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


@dataclass(frozen=True)
class Frame:
    """The run, indexed the way the figures ask for it."""

    records: list[Record]
    models: list[str]
    datasets: list[str]
    sizes: list[int]
    seeds: list[int]
    classes: dict[str, int]

    @property
    def largest(self) -> int:
        return max(self.sizes)

    def pick(self, model: str, dataset: str, size: int | None, seed: int | None = None):
        return [
            r
            for r in self.records
            if (r.model, r.dataset, r.calib_size) == (model, dataset, size)
            and (seed is None or r.seed == seed)
        ]

    def metric(self, model: str, dataset: str, size: int | None, name: str) -> np.ndarray:
        """One value per seed (a single value for raw), seeds in order."""
        rows = sorted(self.pick(model, dataset, size), key=lambda r: r.seed or 0)
        return np.array([r.metrics[name] for r in rows])

    def pooled(self, model: str, size: int | None) -> Bins:
        rows = [r.bins for d in self.datasets for r in self.pick(model, d, size)]
        return sum(rows[1:], rows[0])


def load(path: Path) -> Frame:
    results = Results.load(path)
    present = {(r.model, r.dataset) for r in results.records}
    config = results.config
    return Frame(
        records=results.records,
        models=[m for m in config.models if any(m == p for p, _ in present)],
        datasets=[d for d in config.datasets if any(d == q for _, q in present)],
        sizes=sorted(config.experiment.calib_sizes),
        seeds=list(config.experiment.seeds),
        classes={key: len(spec.options) for key, spec in config.datasets.items()},
    )


def name(model: str) -> str:
    return DISPLAY.get(model, model)


def colours(frame: Frame) -> dict[str, str]:
    return {m: MODEL_COLOURS[i % len(MODEL_COLOURS)] for i, m in enumerate(frame.models)}


# ------------------------------------------------------------------ numbers ----


def ece(bins: Bins) -> float:
    """Σ (count_b / N) · |acc_b − conf_b| over the summed bins."""
    count = np.asarray(bins.count, float)
    keep = count > 0
    gap = np.abs(np.asarray(bins.correct)[keep] - np.asarray(bins.confidence)[keep])
    return float(gap.sum() / count.sum())


def selective(bins: Bins, threshold: float) -> tuple[float, float, float]:
    """Coverage, actual accuracy and promised accuracy over bins at or above a threshold.

    Thresholds are snapped to the nearest bin edge: the records keep bins, not examples.
    """
    edges = np.asarray(bins.edges[:-1])
    start = int(np.argmin(np.abs(edges - threshold)))
    count = np.asarray(bins.count, float)
    kept = count[start:].sum()
    if kept == 0:
        return 0.0, float("nan"), float("nan")
    return (
        kept / count.sum(),
        float(np.sum(bins.correct[start:]) / kept),
        float(np.sum(bins.confidence[start:]) / kept),
    )


def mean_over_datasets(frame: Frame, model: str, size: int | None, metric: str) -> np.ndarray:
    """Per seed, the plain mean over datasets. NLL is taken relative to raw."""
    rows = []
    for dataset in frame.datasets:
        values = frame.metric(model, dataset, size, metric)
        if metric == "nll":
            values = values / frame.metric(model, dataset, None, "nll")[0]
        rows.append(values)
    return np.mean(rows, axis=0)


def stable_from(frame: Frame) -> int | None:
    """Smallest n from which every seed beats raw NLL on every pair that raw left miscalibrated."""
    pairs = [
        (m, d)
        for m in frame.models
        for d in frame.datasets
        if frame.metric(m, d, None, "ece")[0] >= 0.05
    ]

    def holds(size: int) -> bool:
        return all(
            (frame.metric(m, d, size, "nll") < frame.metric(m, d, None, "nll")[0]).all()
            for m, d in pairs
        )

    for index, size in enumerate(frame.sizes):
        if all(holds(s) for s in frame.sizes[index:]):
            return size
    return None


# ------------------------------------------------------------------ figures ----


def fig_reliability(frame: Frame, out: Path) -> None:
    """Hero: raw and calibrated on one reliability diagram per model, all datasets pooled."""
    n = len(frame.models)
    figure, axes = plt.subplots(1, n, figsize=(4.5 * n, 5.0), sharey=True, layout="constrained")
    for ax, model in zip(np.atleast_1d(axes), frame.models, strict=True):
        ax.plot([0, 1], [0, 1], color=INK, ls=(0, (4, 3)), lw=1.2, zorder=2)
        for index, (size, colour, label) in enumerate(
            ((None, RAW, "raw"), (frame.largest, CAL, "logitly"))
        ):
            bins = frame.pooled(model, size)
            count = np.asarray(bins.count, float)
            share = count / count.sum()
            keep = share >= FAINT_SHARE
            conf = np.asarray(bins.confidence)[keep] / count[keep]
            acc = np.asarray(bins.correct)[keep] / count[keep]
            # 95% interval on the accuracy; calibrated bins hold each test example once per seed
            draws = count[keep] / (1 if size is None else len(frame.seeds))
            half = 1.96 * np.sqrt(acc * (1 - acc) / draws)
            ax.fill_between(conf, acc - half, acc + half, color=colour, alpha=0.15, lw=0, zorder=3)
            ax.plot(conf, acc, color=colour, lw=2.2, marker="o", ms=5, zorder=4, clip_on=False)

            edges = np.asarray(bins.edges)
            width = np.diff(edges)
            ax.bar(
                edges[:-1] + width * (0.08 + 0.42 * index),
                share * 0.3,
                width=width * 0.42,
                align="edge",
                color=colour,
                alpha=0.35,
                lw=0,
                zorder=1,
            )
            ax.text(
                0.04,
                0.96 - 0.075 * index,
                f"{label} ECE {mean_of(frame, model, size, 'ece'):.3f}",
                transform=ax.transAxes,
                va="top",
                fontsize=11,
                fontweight="bold",
                color=colour,
            )
        ax.set(xlim=(0, 1), ylim=(0, 1), title=name(model))
        ax.set_aspect("equal")
        ax.set_xlabel("model confidence")
    np.atleast_1d(axes)[0].set_ylabel("actual accuracy")
    handles = [
        Line2D([], [], color=RAW, lw=2.2, marker="o", ms=5, label="raw logits"),
        Line2D(
            [],
            [],
            color=CAL,
            lw=2.2,
            marker="o",
            ms=5,
            label=f"calibrated with logitly ({frame.largest} examples)",
        ),
        Line2D([], [], color=INK, ls=(0, (4, 3)), lw=1.2, label="perfect calibration"),
        Patch(color="#8C959F", alpha=0.5, label="share of answers per bin"),
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=10)
    save(figure, out / "reliability.png")


def fig_data_efficiency(frame: Frame, out: Path) -> int | None:
    """How many labelled examples calibration needs, averaged over datasets."""
    palette = colours(frame)
    sizes = frame.sizes
    knee = stable_from(frame)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    ece_ax, nll_ax, win_ax = axes

    for model in frame.models:
        colour = palette[model]
        for ax, metric in ((ece_ax, "ece"), (nll_ax, "nll")):
            per_size = np.array([mean_over_datasets(frame, model, s, metric) for s in sizes])
            mean, spread = per_size.mean(axis=1), per_size.std(axis=1, ddof=1)
            ax.fill_between(sizes, mean - spread, mean + spread, color=colour, alpha=0.14, lw=0)
            ax.plot(sizes, mean, color=colour, marker="o", ms=5, lw=2, label=name(model))
            if metric == "ece":
                raw = np.mean([frame.metric(model, d, None, "ece")[0] for d in frame.datasets])
                ax.axhline(raw, color=colour, ls=(0, (4, 3)), lw=1.2, alpha=0.8)

    nll_ax.axhline(1, color=INK, ls=(0, (4, 3)), lw=1.2)
    nll_ax.text(sizes[-1], 1.012, "raw", ha="right", va="bottom", fontsize=8.5, color=INK)

    # Share of model × dataset pairs where every seed beats raw.
    pairs = [(m, d) for m in frame.models for d in frame.datasets]
    for metric, colour, label in (("nll", CAL, "NLL"), ("ece", "#8C959F", "ECE")):
        wins = [
            np.mean(
                [
                    (frame.metric(m, d, s, metric) < frame.metric(m, d, None, metric)[0]).all()
                    for m, d in pairs
                ]
            )
            for s in sizes
        ]
        win_ax.plot(
            sizes,
            wins,
            marker="o",
            ms=5,
            lw=2,
            color=colour,
            label=f"{label} better than raw on all {len(frame.seeds)} seeds",
        )
    win_ax.set_ylim(0, 1.02)
    win_ax.set_yticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])

    titles = (
        "Calibration error (ECE), mean of datasets",
        "NLL relative to raw (1.0 = no change)",
        f"Pairs where it reliably helps ({len(pairs)} model × dataset)",
    )
    for ax, title in zip(axes, titles, strict=True):
        ax.set_xscale("log")
        ax.set_xticks(sizes, [str(s) for s in sizes])
        ax.minorticks_off()
        ax.set_xlabel("labelled examples used for calibration")
        ax.set_title(title, fontsize=11)
        if knee is not None:
            ax.axvline(knee, color=INK, lw=1, alpha=0.35)
            ax.axvspan(knee, sizes[-1] * 1.15, color="#2DA44E", alpha=0.05, lw=0)
    ece_ax.set_ylim(bottom=0)
    nll_ax.set_ylim(bottom=0)
    if knee is not None:
        nll_ax.text(knee * 1.06, 0.04, f"n ≥ {knee}: stable gains", fontsize=9, color="#1A7F37")
    handles, labels = ece_ax.get_legend_handles_labels()
    handles.append(Line2D([], [], color=MUTED, ls=(0, (4, 3)), lw=1.2))
    labels.append("raw (same colour)")
    ece_ax.legend(handles, labels, loc="upper right", bbox_to_anchor=(1, 0.9), fontsize=9)
    misses = [
        f"{name(m)} · {d}"
        for m, d in pairs
        if not (frame.metric(m, d, sizes[-1], "nll") < frame.metric(m, d, None, "nll")[0]).all()
    ]
    if misses:
        listed = "\n".join(misses)
        win_ax.text(
            sizes[0],
            0.97,
            f"NLL misses at n = {sizes[-1]}:\n{listed}\n(raw was already calibrated)",
            fontsize=8.5,
            color=MUTED,
            va="top",
        )
    win_ax.legend(loc="lower right", fontsize=9)
    save(figure, out / "data_efficiency.png")
    return knee


def fig_ece_by_dataset(frame: Frame, out: Path) -> None:
    """Small multiples: ECE against calibration size, each dataset on its own scale."""
    palette = colours(frame)
    sizes = frame.sizes
    columns = 3
    rows = -(-len(frame.datasets) // columns)
    figure, axes = plt.subplots(rows, columns, figsize=(13, 3.4 * rows), layout="constrained")
    for ax, dataset in zip(axes.flat, frame.datasets, strict=False):
        for model in frame.models:
            colour = palette[model]
            values = np.array([frame.metric(model, dataset, s, "ece") for s in sizes])
            mean, spread = values.mean(axis=1), values.std(axis=1, ddof=1)
            ax.fill_between(sizes, mean - spread, mean + spread, color=colour, alpha=0.13, lw=0)
            ax.plot(sizes, mean, color=colour, marker="o", ms=4, lw=1.8, label=name(model))
            ax.axhline(
                frame.metric(model, dataset, None, "ece")[0],
                color=colour,
                ls=(0, (4, 3)),
                lw=1.1,
                alpha=0.85,
            )
        ax.set_title(f"{dataset}  ·  {frame.classes[dataset]} classes", fontsize=10.5)
        ax.set_xscale("log")
        ax.set_xticks(sizes, [str(s) for s in sizes])
        ax.minorticks_off()
        ax.set_ylim(bottom=0)
        ax.set_ylabel("ECE")
    for ax in axes.flat[len(frame.datasets) :]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("labelled examples used for calibration")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color=MUTED, ls=(0, (4, 3)), lw=1.1))
    labels.append("raw (same colour)")
    figure.legend(handles, labels, loc="outside lower center", ncol=len(labels), fontsize=9.5)
    figure.suptitle(
        "ECE by dataset; band = spread over 5 calibration draws",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    save(figure, out / "ece_by_dataset.png")


def fig_before_after(frame: Frame, out: Path) -> None:
    """Dumbbells: ECE raw → calibrated for every model × dataset pair."""
    n = len(frame.models)
    figure, axes = plt.subplots(
        1, n, figsize=(4.4 * n, 3.9), sharey=True, sharex=True, layout="constrained"
    )
    datasets = frame.datasets[::-1]
    y = np.arange(len(datasets))
    top = max(frame.metric(m, d, None, "ece")[0] for m in frame.models for d in datasets)
    for ax, model in zip(np.atleast_1d(axes), frame.models, strict=True):
        for row, dataset in zip(y, datasets, strict=True):
            raw = frame.metric(model, dataset, None, "ece")[0]
            cal = frame.metric(model, dataset, frame.largest, "ece").mean()
            better = cal < raw
            ax.annotate(
                "",
                xy=(cal, row),
                xytext=(raw, row),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": CAL if better else "#8C959F",
                    "lw": 1.6,
                    "shrinkA": 5,
                    "shrinkB": 5,
                    "mutation_scale": 11,
                },
            )
            ax.scatter([raw], [row], s=46, color=RAW, zorder=3)
            ax.scatter([cal], [row], s=46, color=CAL, zorder=3)
            change = cal / raw - 1
            ax.text(
                max(raw, cal) + top * 0.035,
                row,
                f"{change:+.0%}",
                va="center",
                fontsize=8.5,
                color=CAL if better else MUTED,
                fontweight="bold" if change < -0.5 else "normal",
            )
        ax.set_title(name(model), fontsize=11)
        ax.set_xlabel("ECE (lower is better)")
        ax.set_xlim(0, top * 1.28)
        ax.grid(axis="y", visible=False)
    first = np.atleast_1d(axes)[0]
    first.set_yticks(y, [f"{d} ({frame.classes[d]})" for d in datasets])
    first.set_ylabel("dataset (classes)")
    handles = [
        Line2D([], [], color=RAW, marker="o", ls="", label="raw"),
        Line2D(
            [],
            [],
            color=CAL,
            marker="o",
            ls="",
            label=f"calibrated on {frame.largest} (mean of {len(frame.seeds)} draws)",
        ),
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=2, fontsize=9.5)
    figure.suptitle(
        "Calibration error before and after, every model and dataset",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    save(figure, out / "before_after.png")


def fig_selective(frame: Frame, out: Path) -> None:
    """Automate the confident share: what a confidence threshold promises, and what it delivers."""
    n = len(frame.models)
    figure, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.4), sharey=True, layout="constrained")
    edges = np.asarray(frame.pooled(frame.models[0], None).edges[:-1])
    cuts = edges[edges >= 0.5]
    for ax, model in zip(np.atleast_1d(axes), frame.models, strict=True):
        for size, colour in ((None, RAW), (frame.largest, CAL)):
            bins = frame.pooled(model, size)
            points = np.array([selective(bins, t) for t in cuts])
            coverage, actual, promised = points.T
            ax.plot(coverage, promised, color=colour, ls=(0, (2, 2)), lw=1.4)
            ax.plot(coverage, actual, color=colour, lw=2.2, marker="o", ms=3.5)
            ax.fill_between(coverage, actual, promised, color=colour, alpha=0.08, lw=0)
            cover, hit, claim = selective(bins, THRESHOLDS[0])
            ax.scatter([cover], [hit], s=70, color=colour, edgecolor="white", lw=1.2, zorder=4)
            ax.annotate(
                f"confidence ≥ {THRESHOLDS[0]:g}: {cover:.0%} handled\n"
                f"promised {claim:.0%}, got {hit:.0%}",
                (cover, hit),
                textcoords="offset points",
                xytext=(-10, -34),
                ha="right",
                fontsize=8.5,
                color=colour,
                fontweight="bold",
            )
        ax.set_title(name(model), fontsize=11)
        ax.set_xlabel("share of inputs the model handles on its own")
        ax.set_xlim(0, 1.02)
        ax.set_xticks(np.linspace(0, 1, 6), [f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    first = np.atleast_1d(axes)[0]
    first.set_ylabel("accuracy on the handled inputs")
    first.set_ylim(0.7, 1.0)
    first.set_yticks(np.linspace(0.7, 1, 7), [f"{v:.0%}" for v in np.linspace(0.7, 1, 7)])
    handles = [
        Line2D([], [], color=INK, lw=2.2, label="actual accuracy"),
        Line2D(
            [], [], color=INK, lw=1.4, ls=(0, (2, 2)), label="promised (mean stated confidence)"
        ),
        Patch(color=RAW, label="raw"),
        Patch(color=CAL, label=f"calibrated on {frame.largest}"),
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=9.5)
    figure.suptitle(
        "Auto-accept above a confidence threshold: raw over-promises, calibrated keeps its word",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    save(figure, out / "selective.png")


def mean_of(frame: Frame, model: str, size: int | None, metric: str) -> float:
    """Mean over datasets of the mean over seeds."""
    return float(np.mean([frame.metric(model, d, size, metric).mean() for d in frame.datasets]))


def fig_simple_ece(frame: Frame, out: Path) -> None:
    """Calibration error per model, before and after, as two bars."""
    x = np.arange(len(frame.models))
    raw = np.array([mean_of(frame, m, None, "ece") for m in frame.models])
    cal = np.array([mean_of(frame, m, frame.largest, "ece") for m in frame.models])
    figure, ax = plt.subplots(figsize=(8.2, 4.3), layout="constrained")
    ax.bar(x - 0.2, raw, 0.38, color=RAW, label="raw")
    ax.bar(x + 0.2, cal, 0.38, color=CAL, label=f"with logitly ({frame.largest} labelled examples)")
    for i in range(len(x)):
        ax.text(x[i] - 0.2, raw[i] + 0.004, f"{raw[i]:.3f}", ha="center", fontsize=10)
        ax.text(x[i] + 0.2, cal[i] + 0.004, f"{cal[i]:.3f}", ha="center", fontsize=10)
        ax.text(
            x[i] + 0.2,
            cal[i] + 0.03,
            f"−{1 - cal[i] / raw[i]:.0%}",
            ha="center",
            fontsize=16,
            fontweight="bold",
            color=CAL,
        )
    ax.set_xticks(x, [name(m) for m in frame.models], fontsize=10.5, color=INK)
    ax.set_ylim(0, raw.max() * 1.35)
    ax.set_ylabel("calibration error, ECE (lower is better)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title(
        f"Calibration error, average over {len(frame.datasets)} datasets", fontsize=13, pad=10
    )
    save(figure, out / "simple_ece.png")


def fig_simple_data(frame: Frame, out: Path) -> None:
    """How many labelled examples: one line, everything averaged."""
    sizes = frame.sizes
    curve = np.array([np.mean([mean_of(frame, m, s, "ece") for m in frame.models]) for s in sizes])
    raw = np.mean([mean_of(frame, m, None, "ece") for m in frame.models])
    figure, ax = plt.subplots(figsize=(8.2, 4.3), layout="constrained")
    ax.axhline(raw, color=RAW, lw=2, ls=(0, (5, 3)))
    ax.text(
        sizes[-1],
        raw + 0.006,
        f"raw: {raw:.3f}",
        ha="right",
        va="bottom",
        fontsize=10.5,
        color=RAW,
        fontweight="bold",
    )
    ax.fill_between(sizes, curve, raw, color=CAL, alpha=0.08, lw=0)
    ax.plot(sizes, curve, color=CAL, lw=2.6, marker="o", ms=7)
    for s, v in zip(sizes, curve, strict=True):
        ax.text(s, v - 0.012, f"{v:.3f}", ha="center", va="top", fontsize=9.5, color=CAL)
    ax.set_xscale("log")
    ax.set_xticks(sizes, [str(s) for s in sizes])
    ax.minorticks_off()
    ax.set_ylim(0, raw * 1.2)
    ax.set_xlabel("labelled examples used for calibration")
    ax.set_ylabel("calibration error, ECE")
    ax.set_title(
        f"How many labelled examples you need ({len(frame.models)} models × "
        f"{len(frame.datasets)} datasets)",
        fontsize=13,
        pad=10,
    )
    save(figure, out / "simple_data.png")


def save(figure, path: Path) -> None:
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


# ------------------------------------------------------------------- tables ----


def tables(frame: Frame, knee: int | None) -> list[str]:
    big = frame.largest
    lines = [
        "# Cross-model tables",
        "",
        "Generated by `benchmark/analyze.py` from `results.json`. Do not edit by hand.",
        "",
        f"Calibrated = fitted on {big} examples, mean of {len(frame.seeds)} draws. Pooled ECE sums",
        "the reliability bins of every dataset first; over- and under-confidence on different",
        "datasets can cancel inside a bin, so it reads lower than the mean of per-dataset ECE.",
        "",
        "## Per model",
        "",
        "| model | ECE, mean of datasets | ECE, pooled bins | NLL vs raw | mean accuracy "
        "| Brier vs raw |",
        "|---|---|---|---|---|---|",
    ]
    for model in frame.models:
        raw_ece, cal_ece = ece(frame.pooled(model, None)), ece(frame.pooled(model, big))
        mean_raw = np.mean([frame.metric(model, d, None, "ece")[0] for d in frame.datasets])
        mean_cal = np.mean([frame.metric(model, d, big, "ece").mean() for d in frame.datasets])
        nll = mean_over_datasets(frame, model, big, "nll").mean()
        acc_raw = np.mean([frame.metric(model, d, None, "accuracy")[0] for d in frame.datasets])
        acc_cal = np.mean([frame.metric(model, d, big, "accuracy").mean() for d in frame.datasets])
        brier = np.mean(
            [
                frame.metric(model, d, big, "brier").mean()
                / frame.metric(model, d, None, "brier")[0]
                for d in frame.datasets
            ]
        )
        lines.append(
            f"| {name(model)} | {mean_raw:.3f} → {mean_cal:.3f} ({mean_cal / mean_raw - 1:+.0%}) "
            f"| {raw_ece:.3f} → {cal_ece:.3f} | {nll - 1:+.0%} "
            f"| {acc_raw:.3f} → {acc_cal:.3f} ({(acc_cal - acc_raw) * 100:+.1f} pp) "
            f"| {brier - 1:+.0%} |"
        )

    lines += [
        "",
        "## Auto-accept above a confidence threshold",
        "",
        "Handled = share of test inputs at or above the threshold. Promised = their mean stated",
        "confidence. Actual = their accuracy. Thresholds snap to the 15-bin edges.",
        "",
        "| model | threshold | raw: handled / promised / actual "
        "| calibrated: handled / promised / actual |",
        "|---|---|---|---|",
    ]
    for model in frame.models:
        for threshold in THRESHOLDS:
            cells = []
            for size in (None, big):
                bins = frame.pooled(model, size)
                edge = bins.edges[int(np.argmin(np.abs(np.asarray(bins.edges[:-1]) - threshold)))]
                cover, hit, claim = selective(bins, threshold)
                cells.append(f"{cover:.0%} / {claim:.1%} / **{hit:.1%}**")
            lines.append(f"| {name(model)} | ≥ {edge:.3f} | {cells[0]} | {cells[1]} |")

    for model in frame.models:
        lines += [
            "",
            f"## {name(model)}",
            "",
            "| dataset | ECE | NLL | Brier | accuracy | flipped |",
            "|---|---|---|---|---|---|",
        ]
        for dataset in frame.datasets:
            cells = []
            for metric in ("ece", "nll", "brier", "accuracy"):
                before = frame.metric(model, dataset, None, metric)[0]
                after = frame.metric(model, dataset, big, metric).mean()
                cells.append(f"{before:.3f} → {after:.3f}")
            flipped = frame.metric(model, dataset, big, "changed").mean()
            lines.append(f"| {dataset} | {' | '.join(cells)} | {flipped:.1%} |")

    lines += ["", "## Stability", ""]
    lines.append(
        f"From n = {knee}, every calibration draw beats raw NLL on every pair whose raw ECE was "
        "0.05 or more."
        if knee
        else "No calibration size beat raw on every miscalibrated pair."
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", type=Path, nargs="?", help="default: from --config")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    path = args.results or Layout(BenchConfig.load(args.config).output_dir).results
    frame = load(path)
    out = path.parent / "analysis"
    out.mkdir(parents=True, exist_ok=True)

    fig_reliability(frame, out)
    knee = fig_data_efficiency(frame, out)
    fig_ece_by_dataset(frame, out)
    fig_before_after(frame, out)
    fig_selective(frame, out)
    fig_simple_ece(frame, out)
    fig_simple_data(frame, out)
    (out / "tables.md").write_text("\n".join(tables(frame, knee)) + "\n", encoding="utf-8")
    print(f"wrote figures and tables.md to {out}")


if __name__ == "__main__":
    main()
