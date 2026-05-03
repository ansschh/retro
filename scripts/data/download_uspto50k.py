"""Download the canonical Schneider USPTO-50K split from RetroSim's GitHub mirror.

Writes train/val/test parquets to `data/uspto50k/` and records the source SHA-256
to `data/uspto50k/uspto50k_sha256.json` so it can be pinned into protocol.yaml.

Source: connorcoley/retrosim/master/retrosim/data/data_processed.csv
       (the canonical Schneider 50K split with class labels and train/valid/test markers)
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import pandas as pd

URL = (
    "https://raw.githubusercontent.com/connorcoley/retrosim/"
    "master/retrosim/data/data_processed.csv"
)


def _split_rxn_smiles(s: str) -> tuple[str, str]:
    """Split atom-mapped 'reactants>reagents>products' into (reactants, products).

    Reagents (the middle field) are folded into reactants since some retrosynthesis
    splits don't separate them. If the field is empty the result is unchanged.
    """
    parts = s.split(">")
    if len(parts) == 3:
        reactants, reagents, products = parts
        if reagents.strip():
            reactants = f"{reactants}.{reagents}" if reactants else reagents
        return reactants, products
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError(f"unexpected rxn_smiles format: {s!r}")


def main() -> None:
    out_dir = Path(__file__).resolve().parents[2] / "data" / "uspto50k"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / "data_processed.csv"

    if not raw.exists():
        print(f"downloading {URL}")
        urllib.request.urlretrieve(URL, raw)

    sha = hashlib.sha256(raw.read_bytes()).hexdigest()
    print(f"sha256: {sha}")
    print(f"size: {raw.stat().st_size} bytes")

    df = pd.read_csv(raw)
    print(f"columns: {list(df.columns)}")
    print(f"rows: {len(df)}")

    if "rxn_smiles" in df.columns:
        rp = df["rxn_smiles"].apply(_split_rxn_smiles)
        df["reactants"] = rp.apply(lambda t: t[0])
        df["product"] = rp.apply(lambda t: t[1])
    elif "reactants" in df.columns and "products" in df.columns:
        df["product"] = df["products"]
    else:
        raise RuntimeError(
            f"unrecognized columns; expected rxn_smiles or reactants+products. "
            f"Got: {list(df.columns)}"
        )

    df["rxn_id"] = (
        df["id"].astype(str) if "id" in df.columns else df.index.astype(str)
    )

    if "set" in df.columns:
        split_col = "set"
    elif "split" in df.columns:
        split_col = "split"
    else:
        raise RuntimeError(
            f"no train/valid/test split column found; expected 'set' or 'split'. "
            f"Got: {list(df.columns)}"
        )

    written = {}
    for paper_split, our_split in [
        ("train", "train"),
        ("valid", "val"),
        ("validation", "val"),
        ("test", "test"),
    ]:
        sub = df[df[split_col] == paper_split][["rxn_id", "product", "reactants"]]
        if not len(sub):
            continue
        out = out_dir / f"{our_split}.parquet"
        sub.to_parquet(out, index=False)
        written[our_split] = len(sub)
        print(f"  {our_split}: {len(sub)} rows -> {out}")

    sha_file = out_dir / "uspto50k_sha256.json"
    sha_file.write_text(
        json.dumps(
            {"data_processed.csv": sha, "url": URL, "splits": written}, indent=2
        )
    )
    print(f"wrote {sha_file}")
    print()
    print("Pin into configs/protocol.yaml under single_step.uspto_50k.split_sha256")


if __name__ == "__main__":
    main()
