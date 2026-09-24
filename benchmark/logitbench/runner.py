"""The sweep: score once, then fit and measure for every size and seed."""

import sys
from collections.abc import Callable, Iterator
from functools import lru_cache, partial

import numpy as np

from logitbench.config import BenchConfig, ExperimentSpec
from logitbench.data import Task, load_task
from logitbench.layout import Layout
from logitbench.metrics import evaluate, reliability
from logitbench.records import Record
from logitbench.sampling import sample
from logitbench.scores import ScoreStore
from logitly import Logitly
from logitly.calibration import fit_calibration, softmax


def run(config: BenchConfig, layout: Layout) -> list[Record]:
    """Every model against every dataset. Models load only when scores are missing."""
    tasks = {key: load_task(key, spec, config.experiment) for key, spec in config.datasets.items()}
    stores = {key: ScoreStore(layout.scores(key)) for key in tasks}

    records: list[Record] = []
    for model_key, model_spec in config.models.items():
        build = lru_cache(1)(model_spec.build)  # weights load once, and only on a cache miss
        for dataset_key, task in tasks.items():
            _log(f"{model_key} / {dataset_key}: {len(task.test.inputs)} test examples")
            signature = {
                "model": model_spec.model_dump(mode="json"),
                "dataset": config.datasets[dataset_key].model_dump(mode="json"),
                "draw": [
                    config.experiment.test_size,
                    config.experiment.calib_pool,
                    config.experiment.data_seed,
                ],
            }
            scores = stores[dataset_key].get(model_key, signature, partial(_score, build, task))
            records += list(measure(model_key, task, scores, config.experiment))
    return records


def measure(
    model_key: str, task: Task, scores: dict[str, np.ndarray], experiment: ExperimentSpec
) -> Iterator[Record]:
    """Raw metrics, then one calibrated run per (size, seed)."""
    test, pool = scores["test"], scores["pool"]
    raw = softmax(test)
    baseline = raw.argmax(axis=1)
    row = partial(
        Record,
        model=model_key,
        dataset=task.key,
        n_test=len(task.test.labels),
    )

    yield row(
        method="raw",
        metrics=evaluate(raw, task.test.labels, baseline=baseline, n_bins=experiment.n_bins),
        bins=reliability(raw, task.test.labels, experiment.n_bins),
    )

    for size in experiment.calib_sizes:
        for seed in experiment.seeds:
            rng = np.random.default_rng([experiment.data_seed, seed, size])
            rows = sample(task.pool.labels, size, rng, balanced=experiment.balanced_calibration)
            fit = fit_calibration(
                pool[rows],
                task.pool.labels[rows],
                options=task.question.options,
                model=model_key,
                config=experiment.fit,
            )
            probs = fit.calibration.apply(test)
            yield row(
                method="calibrated",
                calib_size=size,
                seed=seed,
                temperature=fit.calibration.temperature,
                bias=fit.calibration.bias,
                metrics=evaluate(
                    probs, task.test.labels, baseline=baseline, n_bins=experiment.n_bins
                ),
                bins=reliability(probs, task.test.labels, experiment.n_bins),
            )


def _score(build: Callable[[], Logitly], task: Task) -> dict[str, np.ndarray]:
    llm = build()
    return {
        "pool": llm.scores(task.question, task.pool.inputs, progress=True),
        "test": llm.scores(task.question, task.test.inputs, progress=True),
    }


def _log(message: str) -> None:
    print(f"[bench] {message}", file=sys.stderr, flush=True)
