"""Leaderboard writer tests."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from harness.leaderboard import (
    LEADERBOARD_COLUMNS,
    summarize_predictions,
    write_leaderboard_row,
)


def _write_predictions(tmp_path: Path) -> Path:
    p = tmp_path / "preds.jsonl"
    rows = [
        {
            "rxn_id": "r1",
            "product": "CCO",
            "gold_reactants": "CC(=O)O.CCO",
            "predictions": ["CC(=O)O.CCO", "CCC", "CCN"],
        },
        {
            "rxn_id": "r2",
            "product": "CCN",
            "gold_reactants": "CCBr.N",
            "predictions": ["CCC", "CCBr.N", "CCO"],
        },
        {
            "rxn_id": "r3",
            "product": "Brc1ccccc1",
            "gold_reactants": "c1ccccc1.Br",
            "predictions": ["CCC", "CCN", "CCO"],
        },
    ]
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


def test_summarize_predictions(tmp_path):
    pp = _write_predictions(tmp_path)
    m = summarize_predictions(pp)
    assert m["n_test"] == 3
    # r1 matches at top-1, r2 matches at top-2, r3 doesn't match in top-3
    assert abs(m["top_1"] - 1 / 3) < 1e-9
    assert abs(m["top_3"] - 2 / 3) < 1e-9
    assert m["invalid_smiles_rate"] == 0.0


def test_write_leaderboard_row_creates_header_and_row(tmp_path):
    pp = _write_predictions(tmp_path)
    m = summarize_predictions(pp)
    lb = tmp_path / "leaderboard.csv"
    write_leaderboard_row(
        lb,
        model_name="testmodel",
        model_checkpoint_sha="abc123",
        benchmark_name="testbench",
        benchmark_split_sha="def456",
        metrics=m,
        wallclock_seconds=1.23,
        hardware="L40S",
        git_sha="cafe",
    )
    assert lb.exists()
    with open(lb) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["model_name"] == "testmodel"
    assert rows[0]["benchmark_name"] == "testbench"
    assert rows[0]["hardware"] == "L40S"
    assert rows[0]["git_sha"] == "cafe"
    assert set(reader.fieldnames) == set(LEADERBOARD_COLUMNS)


def test_write_leaderboard_row_appends(tmp_path):
    pp = _write_predictions(tmp_path)
    m = summarize_predictions(pp)
    lb = tmp_path / "leaderboard.csv"
    for name in ["modelA", "modelB", "modelC"]:
        write_leaderboard_row(
            lb,
            model_name=name,
            model_checkpoint_sha="x",
            benchmark_name="b",
            benchmark_split_sha="s",
            metrics=m,
            wallclock_seconds=0.0,
            hardware="cpu",
            git_sha="x",
        )
    with open(lb) as f:
        rows = list(csv.DictReader(f))
    assert [r["model_name"] for r in rows] == ["modelA", "modelB", "modelC"]
