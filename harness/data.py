"""Test split data loading.

Standard schema for any test parquet: columns `rxn_id`, `product`, `reactants`.
Reactants are `.`-joined canonical SMILES.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = ("rxn_id", "product", "reactants")


@dataclass
class ReactionRow:
    rxn_id: str
    product: str
    reactants: str


def load_test_split(parquet_path: str | Path) -> list[ReactionRow]:
    """Load test reactions from a parquet file.

    The file must contain columns rxn_id, product, reactants. Use the scripts
    in `scripts/data/` to generate parquets in this schema.
    """
    df = pd.read_parquet(parquet_path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"missing columns in {parquet_path}: {missing}. "
            f"Required: {REQUIRED_COLUMNS}. Got: {list(df.columns)}"
        )
    return [
        ReactionRow(
            rxn_id=str(row.rxn_id),
            product=str(row.product),
            reactants=str(row.reactants),
        )
        for row in df.itertuples()
    ]
