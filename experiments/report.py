"""Generate the concise thesis report and paired S0/S2 test from experiment outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.stats import holm_bonferroni, mcnemar


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows available."
    columns = [str(column) for column in frame.columns]
    rows = [
        ["" if pd.isna(value) else str(value) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    widths = [max(len(columns[index]), *(len(row[index]) for row in rows)) for index in range(len(columns))]
    header = "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(columns)) + " |"
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def generate_report(results_dir: str | Path = "results") -> Path:
    results_dir = Path(results_dir)
    metrics_path = results_dir / "main_results.csv"
    predictions_path = results_dir / "predictions.csv"
    if not metrics_path.exists() or not predictions_path.exists():
        raise FileNotFoundError("Run `make experiments` before generating the report")
    metrics = pd.read_csv(metrics_path)
    predictions = pd.read_csv(predictions_path)
    comparisons = []
    p_values = {}
    # Paired McNemar tests for the thesis comparisons, Holm-corrected together.
    for left, right in (("S0", "S1"), ("S0", "S2"), ("S1", "S2"), ("S2", "S3")):
        if not {left, right} <= set(predictions["system"]):
            continue
        pair = predictions[predictions["system"].isin([left, right])]
        if "skipped" in pair:
            pair = pair[pair["skipped"].isna()]
        paired = pair.pivot(index="query_id", columns="system", values="correct").dropna()
        if paired.empty or {left, right} - set(paired.columns):
            continue
        b = int(((paired[left] == True) & (paired[right] == False)).sum())
        c = int(((paired[left] == False) & (paired[right] == True)).sum())
        result = mcnemar(b, c)
        name = f"{left}_vs_{right}_accuracy"
        p_values[name] = float(result["p_value"])
        comparisons.append({"comparison": name, "b": b, "c": c, **result})
    adjusted = holm_bonferroni(p_values)
    for row in comparisons:
        row["holm_adjusted_p"] = adjusted.get(row["comparison"], row["p_value"])
    pd.DataFrame(comparisons).to_csv(results_dir / "stats_tests.csv", index=False)

    lines = [
        "# TrustRAG Evaluation Report",
        "",
        "This report was generated from the saved experiment outputs.",
        "",
        "## Main Results",
        "",
        _markdown_table(metrics),
        "",
        "## Statistical Tests",
        "",
        _markdown_table(pd.DataFrame(comparisons)) if comparisons else "No paired comparisons were available.",
        "",
    ]
    benchmark_path = Path("data/benchmark/benchmark.csv")
    if benchmark_path.exists() and "correct" in predictions:
        benchmark = pd.read_csv(benchmark_path, dtype=str)
        if "batch" in benchmark and benchmark["batch"].nunique() > 0:
            merged = predictions.merge(benchmark[["query_id", "batch"]], on="query_id", how="left")
            merged = merged[merged["correct"].notna()]
            by_batch = (
                merged.assign(correct=merged["correct"].astype(bool))
                .pivot_table(index="batch", columns="system", values="correct", aggfunc="mean")
                .round(3)
                .reset_index()
            )
            counts = merged.groupby("batch")["query_id"].nunique().rename("queries").reset_index()
            lines.extend(["## Accuracy by Batch", "", _markdown_table(by_batch.merge(counts, on="batch")), ""])
    manifest_path = results_dir / "run_manifest.json"
    if manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        lines.extend(["## Run Manifest", "", "```json", json.dumps(manifest, indent=2), "```", ""])
        if manifest.get("smoke"):
            lines.extend(["**This is a smoke run, not thesis benchmark evidence.**", ""])
    report_path = results_dir / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    print(generate_report())


if __name__ == "__main__":
    main()