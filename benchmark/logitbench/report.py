"""The Markdown, generated from the records and nothing else."""

from collections.abc import Iterable, Sequence

from logitbench.layout import Layout
from logitbench.metrics import LABELS, LOWER_IS_BETTER
from logitbench.records import Cell, Group, Results, aggregate

COLUMNS = ("accuracy", "nll", "ece", "brier", "mean_confidence", "changed")


def write_report(results: Results, layout: Layout) -> list[str]:
    """One index at the root, one report per dataset folder. Returns the datasets written."""
    cells = aggregate(results.records)
    datasets = _ordered(results.config.datasets, {group.dataset for group in cells})
    models = _ordered(results.config.models, {group.model for group in cells})

    for dataset in datasets:
        page = _dataset_page(results, cells, dataset, models)
        _write(layout.dataset_report(dataset), page)
    _write(layout.report, _index(results, cells, datasets, models))
    return datasets


def _index(
    results: Results, cells: dict[Group, dict[str, Cell]], datasets: list[str], models: list[str]
) -> list[str]:
    largest = max(results.config.experiment.calib_sizes)
    return [
        "# logitly calibration benchmark",
        "",
        f"Generated {results.created} by `benchmark/run.py`. Do not edit by hand.",
        "",
        _settings(results),
        "",
        "## Summary",
        "",
        f"Raw vs. calibrated on {largest} examples, averaged over "
        f"{len(results.config.experiment.seeds)} seeds.",
        "",
        _summary(cells, datasets, models, largest),
        "",
        "## Per dataset",
        "",
        *[f"- [{dataset}]({dataset}/report.md)" for dataset in datasets],
    ]


def _dataset_page(
    results: Results, cells: dict[Group, dict[str, Cell]], dataset: str, models: list[str]
) -> list[str]:
    spec = results.config.datasets[dataset]
    lines = [
        f"# {dataset}",
        "",
        f"`{spec.path}` — {spec.instructions} Options: {', '.join(spec.options)}.",
        "",
        f"Test split `{spec.test_split}`, calibration drawn from `{spec.calib_split}`.",
        "",
    ]
    for metric in results.config.report.curve_metrics:
        lines += [f"![{metric} vs calibration size](plots/{metric}.png)", ""]

    for model in models:
        groups = sorted(
            (g for g in cells if g.dataset == dataset and g.model == model),
            key=lambda g: (g.calib_size is not None, g.calib_size or 0),
        )
        if not groups:
            continue
        lines += [f"## {model}", "", _detail(cells, groups), ""]
        if results.config.report.reliability:
            lines += [f"![reliability](plots/reliability_{model}.png)", ""]
    return lines


def _settings(results: Results) -> str:
    experiment = results.config.experiment
    rows = [
        ("models", ", ".join(f"`{key}`" for key in results.config.models)),
        ("test examples", str(experiment.test_size)),
        ("calibration pool", str(experiment.calib_pool)),
        ("calibration sizes", ", ".join(str(size) for size in experiment.calib_sizes)),
        ("seeds", ", ".join(str(seed) for seed in experiment.seeds)),
        ("ECE bins", str(experiment.n_bins)),
        ("calibration draw", "balanced" if experiment.balanced_calibration else "natural mix"),
    ]
    return _table(("setting", "value"), rows)


def _summary(
    cells: dict[Group, dict[str, Cell]], datasets: list[str], models: list[str], largest: int
) -> str:
    rows = []
    for dataset in datasets:
        for model in models:
            raw, fitted = (
                cells.get(Group(model, dataset, None)),
                cells.get(Group(model, dataset, largest)),
            )
            if raw is None or fitted is None:
                continue
            deltas = [
                _delta(raw[m].mean, fitted[m].mean, m) for m in ("accuracy", "nll", "ece", "brier")
            ]
            rows.append([dataset, model, *deltas, _percent(fitted["changed"].mean)])
    return _table(("dataset", "model", "accuracy", "NLL", "ECE", "Brier", "changed"), rows)


def _detail(cells: dict[Group, dict[str, Cell]], groups: Sequence[Group]) -> str:
    header = ("calibration", *(LABELS[metric] for metric in COLUMNS))
    rows = [[group.label, *(_cell(cells[group][m], m) for m in COLUMNS)] for group in groups]
    return _table(header, rows)


def _cell(cell: Cell, metric: str) -> str:
    show = _percent if metric == "changed" else "{:.3f}".format
    return show(cell.mean) if cell.n_seeds < 2 else f"{show(cell.mean)} ± {show(cell.std)}"


def _delta(before: float, after: float, metric: str) -> str:
    change = round(after - before, 3)
    if change == 0:
        return f"{before:.3f} → {after:.3f} (=)"
    mark = " ✓" if (change < 0) == (metric in LOWER_IS_BETTER) else " ✗"
    return f"{before:.3f} → {after:.3f} ({change:+.3f}){mark}"


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    return "\n".join([*lines, *("| " + " | ".join(row) + " |" for row in rows)])


def _ordered(configured: dict[str, object], present: set[str]) -> list[str]:
    """Config order, restricted to what actually ran."""
    return [key for key in configured if key in present]


def _write(path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
