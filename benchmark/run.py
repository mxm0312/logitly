"""Run the logitly calibration benchmark.

    uv run benchmark/run.py
    uv run benchmark/run.py --models qwen2.5-1.5b-instruct --datasets sst2
    uv run benchmark/run.py --config my.yml

Scores are cached per dataset, so a second run only redoes what changed.
"""

import argparse
from pathlib import Path

from logitbench import BenchConfig, Layout, Results, run, write_plots, write_report

DEFAULT_CONFIG = Path(__file__).with_name("config.yml")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="benchmark settings")
    parser.add_argument("--models", nargs="+", default=[], help="config keys to run; default all")
    parser.add_argument("--datasets", nargs="+", default=[], help="config keys to run; default all")
    args = parser.parse_args()

    config = BenchConfig.load(args.config).select(args.models, args.datasets)
    layout = Layout(config.output_dir)

    results = Results.of(config, run(config, layout))
    results.save(layout.results)
    write_plots(results, layout)
    datasets = write_report(results, layout)

    print(f"\n{len(results.records)} records -> {layout.results}")
    print(f"report              -> {layout.report}")
    for dataset in datasets:
        print(f"  {dataset:<12} -> {layout.dataset_report(dataset)}")


if __name__ == "__main__":
    main()
