"""Leaderboard writer.

Reads a JSONL of per-target predictions, computes metrics via the metrics
module, appends one row to results/leaderboard.csv. Every row records
protocol_version + protocol_sha so reruns are traceable.

JSONL schema (one line per test reaction):
  {"rxn_id": str, "product": str, "gold_reactants": str,
   "predictions": [reactant_set_smiles, ...]}
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import click

from metrics import (
    candidate_diversity,
    invalid_smiles_rate,
    load_protocol,
    maxfrag_accuracy,
    protocol_sha,
    top_k_exact_match,
)

LEADERBOARD_COLUMNS = [
    "timestamp_utc",
    "model_name",
    "model_checkpoint_sha",
    "benchmark_name",
    "benchmark_split_sha",
    "protocol_version",
    "protocol_sha",
    "n_test",
    "top_1",
    "top_3",
    "top_5",
    "top_10",
    "invalid_smiles_rate",
    "candidate_diversity_mean",
    "maxfrag_top_1",
    "wallclock_seconds",
    "hardware",
    "git_sha",
    "predictions_path",
]


def summarize_predictions(predictions_jsonl: str | Path) -> dict[str, Any]:
    """Read predictions JSONL and compute aggregate metrics."""
    rows = []
    with open(predictions_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"no rows in {predictions_jsonl}")
    n_test = len(rows)

    top_k_correct = {1: 0, 3: 0, 5: 0, 10: 0}
    maxfrag1 = 0
    diversity_sum = 0.0
    all_predictions: list[str] = []

    for r in rows:
        preds = r["predictions"]
        gold = r["gold_reactants"]
        for k in top_k_correct:
            top_k_correct[k] += top_k_exact_match(preds, gold, k=k)
        maxfrag1 += maxfrag_accuracy(preds, gold, k=1)
        diversity_sum += candidate_diversity(preds)
        all_predictions.extend(preds)

    return {
        "n_test": n_test,
        "top_1": top_k_correct[1] / n_test,
        "top_3": top_k_correct[3] / n_test,
        "top_5": top_k_correct[5] / n_test,
        "top_10": top_k_correct[10] / n_test,
        "maxfrag_top_1": maxfrag1 / n_test,
        "candidate_diversity_mean": diversity_sum / n_test,
        "invalid_smiles_rate": invalid_smiles_rate(all_predictions),
    }


def write_leaderboard_row(
    leaderboard_path: str | Path,
    *,
    model_name: str,
    model_checkpoint_sha: str,
    benchmark_name: str,
    benchmark_split_sha: str,
    metrics: dict[str, Any],
    wallclock_seconds: float,
    hardware: str,
    git_sha: str,
    predictions_path: str = "",
    protocol_path: str | Path | None = None,
) -> None:
    """Append one row to the leaderboard CSV. Creates header if missing."""
    p = load_protocol(protocol_path)
    psha = protocol_sha(protocol_path)

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_name": model_name,
        "model_checkpoint_sha": model_checkpoint_sha,
        "benchmark_name": benchmark_name,
        "benchmark_split_sha": benchmark_split_sha,
        "protocol_version": p.get("protocol_version", "unknown"),
        "protocol_sha": psha[:12],
        "n_test": metrics.get("n_test", ""),
        "top_1": metrics.get("top_1", ""),
        "top_3": metrics.get("top_3", ""),
        "top_5": metrics.get("top_5", ""),
        "top_10": metrics.get("top_10", ""),
        "invalid_smiles_rate": metrics.get("invalid_smiles_rate", ""),
        "candidate_diversity_mean": metrics.get("candidate_diversity_mean", ""),
        "maxfrag_top_1": metrics.get("maxfrag_top_1", ""),
        "wallclock_seconds": wallclock_seconds,
        "hardware": hardware,
        "git_sha": git_sha,
        "predictions_path": predictions_path,
    }

    leaderboard_path = Path(leaderboard_path)
    file_exists = leaderboard_path.exists()
    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)

    with open(leaderboard_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEADERBOARD_COLUMNS)
        if not file_exists:
            w.writeheader()
        w.writerow(row)


@click.command()
@click.option("--predictions", required=True, help="JSONL predictions file")
@click.option("--leaderboard", default="results/leaderboard.csv", show_default=True)
@click.option("--model-name", required=True)
@click.option("--model-checkpoint-sha", default="unknown")
@click.option("--benchmark-name", required=True)
@click.option("--benchmark-split-sha", default="unknown")
@click.option("--wallclock-seconds", type=float, required=True)
@click.option("--hardware", default="unknown")
@click.option("--git-sha", default="unknown")
def main(
    predictions: str,
    leaderboard: str,
    model_name: str,
    model_checkpoint_sha: str,
    benchmark_name: str,
    benchmark_split_sha: str,
    wallclock_seconds: float,
    hardware: str,
    git_sha: str,
) -> None:
    metrics = summarize_predictions(predictions)
    write_leaderboard_row(
        leaderboard,
        model_name=model_name,
        model_checkpoint_sha=model_checkpoint_sha,
        benchmark_name=benchmark_name,
        benchmark_split_sha=benchmark_split_sha,
        metrics=metrics,
        wallclock_seconds=wallclock_seconds,
        hardware=hardware,
        git_sha=git_sha,
        predictions_path=str(predictions),
    )
    click.echo(
        f"appended to {leaderboard}: "
        f"top1={metrics['top_1']:.4f} "
        f"top5={metrics['top_5']:.4f} "
        f"top10={metrics['top_10']:.4f} "
        f"maxfrag1={metrics['maxfrag_top_1']:.4f} "
        f"invalid={metrics['invalid_smiles_rate']:.4f} "
        f"n={metrics['n_test']}"
    )


if __name__ == "__main__":
    main()
