"""Rebuild the tables and plots of an earlier run. No model is loaded.

uv run benchmark/report.py                       # output directory from config.yml
uv run benchmark/report.py --config my.yml
uv run benchmark/report.py path/to/results.json
"""

import argparse
from pathlib import Path

from logitbench import BenchConfig, Layout, Results, write_plots, write_report

DEFAULT_CONFIG = Path(__file__).with_name("config.yml")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results", type=Path, nargs="?", help="default: from --config")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    path = args.results or Layout(BenchConfig.load(args.config).output_dir).results
    results = Results.load(path)
    datasets = write_report(results, Layout(path.parent))
    write_plots(results, Layout(path.parent))
    print(f"rewrote {path.parent / 'report.md'} and {len(datasets)} dataset report(s)")


if __name__ == "__main__":
    main()
