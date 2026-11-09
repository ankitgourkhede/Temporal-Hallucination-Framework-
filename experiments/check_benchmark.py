"""CLI for benchmark schema, split leakage, and nationality coverage checks."""

from __future__ import annotations

import argparse

import yaml

from experiments.benchmark import check_nationality_coverage, validate_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the benchmark CSV")
    parser.add_argument("--benchmark", default="data/benchmark/benchmark.csv")
    parser.add_argument("--classifier-data", default="data/train/queries_labeled.csv")
    args = parser.parse_args()
    frame = validate_benchmark(args.benchmark, args.classifier_data)
    coverage = check_nationality_coverage(frame)
    print(yaml.safe_dump({"queries": len(frame), "splits": frame["split"].value_counts().to_dict(), "non_indian_fraction": coverage}, sort_keys=False))


if __name__ == "__main__":
    main()