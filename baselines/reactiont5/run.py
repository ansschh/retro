"""ReactionT5 v2 retrosynthesis runner.

Loads `sagawa/ReactionT5v2-retrosynthesis-USPTO_50k` from HuggingFace, runs
inference on a test parquet (columns: rxn_id, product, reactants), writes
per-target top-k predictions to a JSONL file. The leaderboard scorer reads it.

Input format verified from model card on 2026-05-02: just the product SMILES,
no prefix template. Output is `.`-joined reactant SMILES; we strip spaces and
trailing dots before writing.

Usage on cluster:
  source env.sh && source .venv/bin/activate
  python -m baselines.reactiont5.run \
      --test-data data/smoke/uspto50k_smoke_test.parquet \
      --output-jsonl results/reactiont5_smoke_topk10.jsonl \
      --topk 10
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import click
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from harness.data import load_test_split

MODEL_NAME = "sagawa/ReactionT5v2-retrosynthesis-USPTO_50k"


def _decode(tokenizer, output_ids) -> str:
    """ReactionT5 tokenizer inserts spaces between SMILES tokens; strip them.
    Also rstrip trailing dots (model sometimes appends them)."""
    s = tokenizer.decode(output_ids, skip_special_tokens=True)
    return s.replace(" ", "").rstrip(".")


@click.command()
@click.option(
    "--test-data",
    required=True,
    help="Path to test parquet (columns: rxn_id, product, reactants)",
)
@click.option("--output-jsonl", required=True, help="Output JSONL path")
@click.option("--topk", default=10, show_default=True, help="num_beams = num_return_sequences")
@click.option("--batch-size", default=16, show_default=True)
@click.option("--max-input-length", default=256, show_default=True)
@click.option("--max-output-length", default=256, show_default=True)
@click.option("--device", default=None, help="cuda or cpu (default: auto)")
@click.option(
    "--limit",
    default=0,
    type=int,
    help="If >0, only run on first N test reactions (smoke testing).",
)
def main(
    test_data: str,
    output_jsonl: str,
    topk: int,
    batch_size: int,
    max_input_length: int,
    max_output_length: int,
    device: str | None,
    limit: int,
) -> None:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    click.echo(f"loading {MODEL_NAME} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device).eval()

    rows = load_test_split(test_data)
    if limit > 0:
        rows = rows[:limit]
    click.echo(f"running inference on {len(rows)} reactions, top-{topk}")

    out_path = Path(output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    n_done = 0
    with open(out_path, "w") as f, torch.inference_mode():
        for batch_start in tqdm(range(0, len(rows), batch_size)):
            batch = rows[batch_start : batch_start + batch_size]
            products = [r.product for r in batch]

            inp = tokenizer(
                products,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_length,
            ).to(device)

            outputs = model.generate(
                **inp,
                num_beams=topk,
                num_return_sequences=topk,
                max_length=max_output_length,
                early_stopping=True,
            )
            outputs = outputs.view(len(batch), topk, -1)

            for i, row in enumerate(batch):
                preds = [_decode(tokenizer, outputs[i, k]) for k in range(topk)]
                f.write(
                    json.dumps(
                        {
                            "rxn_id": row.rxn_id,
                            "product": row.product,
                            "gold_reactants": row.reactants,
                            "predictions": preds,
                        }
                    )
                    + "\n"
                )
                n_done += 1

    elapsed = time.time() - t_start
    click.echo(
        f"done: {n_done} reactions in {elapsed:.1f}s "
        f"({n_done / elapsed:.2f} rxn/s); wrote {out_path}"
    )


if __name__ == "__main__":
    main()
