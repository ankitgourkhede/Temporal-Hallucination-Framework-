"""Freshness-weight sensitivity study on development data (train + dev splits only).

Question: does the best freshness weight differ by category? If it does, no single fixed
weight vector (S1) can be optimal for every category, which is the case for category-adaptive
weighting (S2). The test split is never read here.

Method
  1. Run each development query once through the real pipeline and keep its scored evidence.
     Retrieval and trust features (A, F, G, QE, SR, CP) do not depend on the weights.
  2. Replay trust scoring and slot fusion offline for each weight vector in a sweep.
     The replay is checked against the real pipeline for S1 and S2 before sweeping.
  3. Sweep w_F per category. The other positive weights keep S1's proportions and are rescaled
     so the positive weights still sum to 1; w_C stays at S1's value.

Metrics per (category, w_F)
  accuracy    fused value for the gold slot equals the gold value
  gold_share  support for the gold value / total support (decision margin; 1.0 = unanimous)
  auroc       how well evidence Trust ranks correct above incorrect values

    PYTHONPATH=. python -m experiments.sensitivity --config trustrag/config.benchmark.yaml --label content_relative
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from experiments.metrics import values_match
from experiments.run_all import benchmark_network_guard
from trustrag.online.policy.weights import load_weights
from trustrag.online.trust.fusion import fuse_slots
from trustrag.online.trust.scoring import _same_value
from trustrag.pipeline import TrustRAGPipeline

DEV_SPLITS = ("train", "dev")
W_F_GRID = [round(value, 2) for value in np.arange(0.0, 0.61, 0.05)]
POSITIVE = ("w_A", "w_F", "w_G", "w_E", "w_SR")


def collect(config_path: str, benchmark_path: str) -> list[dict[str, Any]]:
    """Run development queries once (S1 and S2) and keep gold-slot evidence features."""
    frame = pd.read_csv(benchmark_path, dtype=str).fillna("")
    if not set(frame["split"]) >= set(DEV_SPLITS):
        raise ValueError("Benchmark must contain train and dev splits")
    development = frame[frame["split"].isin(DEV_SPLITS)]
    pipeline = TrustRAGPipeline(config_path)
    records = []
    with benchmark_network_guard():
        for row in development.to_dict("records"):
            resolved = {}
            evidence = None
            for system in ("S1", "S2"):
                response = pipeline.answer(row["query"], system=system, llm="mock", as_of=row["as_of_date"], mode="benchmark")
                trace = response["trace"]
                resolved[system] = next(
                    (item.get("value") for item in trace.get("resolved_slots", []) if item.get("slot") == row["gold_slot"]),
                    None,
                )
                if evidence is None:
                    evidence = [
                        item for item in trace.get("trust", {}).get("evidence", [])
                        if item.get("slot") == row["gold_slot"]
                    ]
            records.append({"row": row, "evidence": evidence or [], "pipeline": resolved})
    return records


def trust_of(item: dict[str, Any], weights: dict[str, float]) -> float:
    """The trust formula from trustrag.online.trust.scoring, conflict_mode='both'."""
    positive = (
        weights["w_A"] * item["A"] + weights["w_F"] * item["F"] + weights["w_G"] * item["G"]
        + weights["w_E"] * item["QE"] + weights["w_SR"] * item["SR"]
    )
    raw = positive - weights["w_C"] * item["CP"]
    return min(1.0, max(0.0, min(1.0, max(0.0, raw)) * (1.0 - item["CP"])))


def replay(record: dict[str, Any], weights: dict[str, float]) -> dict[str, Any]:
    evidence = [{**item, "Trust": trust_of(item, weights)} for item in record["evidence"]]
    gold = record["row"]["gold_value"]
    if not evidence:
        return {"value": None, "correct": False, "gold_share": 0.0, "labels": [], "scores": []}
    fused = fuse_slots(evidence)
    value = fused[0]["value"] if fused else None
    # Support per distinct value, counting each domain once (as fuse_slots does).
    groups: list[tuple[Any, dict[str, float]]] = []
    for item in evidence:
        group = next((g for v, g in groups if _same_value(item.get("value_norm"), v)), None)
        if group is None:
            group = {}
            groups.append((item.get("value_norm"), group))
        domain = str(item.get("source_domain"))
        group[domain] = max(group.get(domain, 0.0), item["Trust"])
    support = [(v, sum(g.values())) for v, g in groups]
    total = sum(amount for _, amount in support)
    gold_support = sum(amount for v, amount in support if values_match(v, gold))
    return {
        "value": value,
        "correct": values_match(value, gold),
        "gold_share": gold_support / total if total else 0.0,
        "labels": [values_match(item.get("value_norm"), gold) for item in evidence],
        "scores": [item["Trust"] for item in evidence],
    }


def vector_with_freshness(base: dict[str, float], w_f: float) -> dict[str, float]:
    others = [key for key in POSITIVE if key != "w_F"]
    scale = (1.0 - w_f) / sum(base[key] for key in others)
    vector = {key: base[key] * scale for key in others}
    vector.update({"w_F": w_f, "w_C": base["w_C"]})
    return vector


def summarize(records: list[dict[str, Any]], weights: dict[str, float]) -> dict[str, float]:
    results = [replay(record, weights) for record in records]
    labels = [label for result in results for label in result["labels"]]
    scores = [score for result in results for score in result["scores"]]
    auroc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else float("nan")
    return {
        "n_queries": len(results),
        "accuracy": float(np.mean([result["correct"] for result in results])),
        "gold_share": float(np.mean([result["gold_share"] for result in results])),
        "auroc": float(auroc),
    }


def check_replay(records: list[dict[str, Any]], config_path: str) -> dict[str, int]:
    mismatches = {}
    for system, mode in (("S1", "static"), ("S2", "policy")):
        count = 0
        for record in records:
            if not record["evidence"]:
                continue
            weights, _ = load_weights(config_path, record["row"]["category"], mode)
            replayed = replay(record, weights)["value"]
            if not _same_value(replayed, record["pipeline"][system]) and replayed != record["pipeline"][system]:
                count += 1
        mismatches[system] = count
    return mismatches


def plot(sweep: pd.DataFrame, s2_points: dict[str, float], s1_w_f: float, output: Path, label: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"visa_policy": "#2a78d6", "vaccination_requirement": "#eb6834", "customs_regulation": "#1baf7a"}
    names = {"visa_policy": "Visa", "vaccination_requirement": "Vaccination", "customs_regulation": "Customs"}
    ink, muted, grid, surface = "#1f1f1e", "#6b6a64", "#e6e5e0", "#fcfcfb"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=surface)
    handles = {}
    for axis, metric, title in ((axes[0], "accuracy", "Accuracy"), (axes[1], "gold_share", "Gold-value support share")):
        axis.set_facecolor(surface)
        ends = []
        for category, frame in sweep.groupby("category"):
            frame = frame.sort_values("w_F")
            (line,) = axis.plot(frame["w_F"], frame[metric], color=colors[category], linewidth=2,
                                solid_capstyle="round", label=names[category])
            handles[names[category]] = line
            ends.append([frame.iloc[-1]["w_F"], frame.iloc[-1][metric], names[category]])
            s2 = frame.iloc[(frame["w_F"] - s2_points[category]).abs().argmin()]
            axis.scatter([s2["w_F"]], [s2[metric]], s=64, color=colors[category], edgecolors=surface, linewidths=2, zorder=3)
        low, high = axis.get_ylim()
        gap = (high - low) * 0.07
        ends.sort(key=lambda end: end[1], reverse=True)
        for index in range(1, len(ends)):
            ends[index][1] = min(ends[index][1], ends[index - 1][1] - gap)
        for x, y, name in ends:
            axis.annotate(name, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=ink,
                          annotation_clip=False)
        axis.axvline(s1_w_f, color=muted, linewidth=1, linestyle=(0, (4, 3)))
        axis.annotate("S1 fixed w_F", (s1_w_f, 0.02), xytext=(4, 0), textcoords="offset points", fontsize=8,
                      color=muted, va="bottom", xycoords=("data", "axes fraction"))
        axis.set_title(title, loc="left", fontsize=11, color=ink)
        axis.set_xlabel("Freshness weight w_F", fontsize=9, color=muted)
        axis.set_xlim(-0.01, 0.72)
        axis.grid(axis="y", color=grid, linewidth=0.8)
        axis.tick_params(colors=muted, labelsize=8)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(grid)
    legend = fig.legend(handles.values(), handles.keys(), loc="upper right", ncol=3, frameon=False, fontsize=9)
    for text in legend.get_texts():
        text.set_color(ink)
    fig.suptitle(f"Freshness-weight sensitivity on development data ({label}); dots = S2 policy weight",
                 x=0.01, ha="left", fontsize=11, color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output, dpi=150, facecolor=surface)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="trustrag/config.benchmark.yaml")
    parser.add_argument("--benchmark", default="data/benchmark/benchmark.csv")
    parser.add_argument("--label", default="content_relative")
    parser.add_argument("--output", default="results/sensitivity")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = collect(args.config, args.benchmark)
    mismatches = check_replay(records, args.config)
    if any(mismatches.values()):
        raise SystemExit(f"Offline replay disagrees with the pipeline: {mismatches}")

    static, _ = load_weights(args.config, "visa_policy", "static")
    policy = yaml.safe_load((Path(args.config).parent / "policies/weights_policy.yaml").read_text(encoding="utf-8"))
    rows = []
    for category in sorted({record["row"]["category"] for record in records}):
        subset = [record for record in records if record["row"]["category"] == category and record["evidence"]]
        for w_f in W_F_GRID:
            rows.append({"category": category, "w_F": w_f, **summarize(subset, vector_with_freshness(static, w_f))})
        for system, weights in (("S1", static), ("S2", {k: float(v) for k, v in policy[category].items()})):
            rows.append({"category": category, "w_F": float(weights["w_F"]), "system": system, **summarize(subset, weights)})
    table = pd.DataFrame(rows)
    sweep = table[table.get("system").isna()] if "system" in table else table
    table.to_csv(output / f"sweep_{args.label}.csv", index=False)
    plot(sweep, {c: float(policy[c]["w_F"]) for c in policy}, float(static["w_F"]),
         output / f"freshness_weight_{args.label}.png", args.label)
    summary = {
        "replay_mismatches": mismatches,
        "queries_with_evidence": int(sum(bool(r["evidence"]) for r in records)),
        "best_w_F_by_gold_share": {
            category: float(frame.loc[frame["gold_share"].idxmax(), "w_F"])
            for category, frame in sweep.groupby("category")
        },
    }
    (output / f"summary_{args.label}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
