"""Generate a hermetic smoke-test dataset of 10 hand-picked reactions.

Used for end-to-end pipeline smoke tests when the real USPTO-50K split isn't
downloaded yet. NOT representative of real USPTO-50K accuracy — the smoke
exercises the pipeline (model load, inference, JSONL write, metrics, leaderboard
row), not the model's true retrosynthesis quality.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem


# (rxn_id, product_smiles, reactants_dot_smiles)
# Picked for SMILES validity, not chemical fidelity. ReactionT5 may or may not
# match these exactly — that's fine; we just want the pipeline to produce
# a meaningful number end to end.
SMOKE_REACTIONS = [
    ("smoke_001", "CCOC(C)=O",                  "CC(=O)O.CCO"),
    ("smoke_002", "c1ccc(Oc2ccccc2)cc1",        "Oc1ccccc1.Brc1ccccc1"),
    ("smoke_003", "CCNC(C)=O",                  "CC(=O)Cl.CCN"),
    ("smoke_004", "Nc1ccccc1",                  "[O-][N+](=O)c1ccccc1"),
    ("smoke_005", "CC(C)O",                     "CC(C)=O"),
    ("smoke_006", "c1ccc(-c2ccccc2)cc1",        "OB(O)c1ccccc1.Brc1ccccc1"),
    ("smoke_007", "O=C(c1ccccc1)c1ccccc1",      "OC(c1ccccc1)c1ccccc1"),
    ("smoke_008", "CC(=O)Nc1ccccc1",            "CC(=O)O.Nc1ccccc1"),
    ("smoke_009", "OCc1ccccc1",                 "O=Cc1ccccc1"),
    ("smoke_010", "Oc1ccc(Cl)cc1",              "Fc1ccc(Cl)cc1.O"),
]


def _canonicalize(s: str) -> str:
    m = Chem.MolFromSmiles(s)
    if m is None or m.GetNumAtoms() == 0:
        raise ValueError(f"invalid SMILES: {s}")
    return Chem.MolToSmiles(m)


def _canonicalize_set(s: str) -> str:
    parts = [_canonicalize(p) for p in s.split(".") if p.strip()]
    return ".".join(sorted(parts))


def main() -> None:
    out_dir = Path(__file__).resolve().parents[2] / "data" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for rxn_id, product, reactants in SMOKE_REACTIONS:
        rows.append(
            {
                "rxn_id": rxn_id,
                "product": _canonicalize(product),
                "reactants": _canonicalize_set(reactants),
            }
        )
    df = pd.DataFrame(rows)
    out_path = out_dir / "uspto50k_smoke_test.parquet"
    df.to_parquet(out_path, index=False)
    print(f"wrote {len(df)} rows to {out_path}")
    print(df.to_string())


if __name__ == "__main__":
    main()
