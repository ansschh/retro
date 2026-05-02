"""Single-step retrosynthesis metrics.

All comparisons go through `metrics.canonicalize`. Predictions and gold
strings are canonicalized identically before equality checks.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Callable, Optional

from rdkit import Chem

from .canonicalize import (
    DEFAULT_POLICY,
    CanonPolicy,
    canon,
    canon_rxn_side,
    is_valid_smiles,
)


def top_k_exact_match(
    predictions: list[str],
    gold: str,
    k: int,
    policy: CanonPolicy = DEFAULT_POLICY,
) -> int:
    """1 if any of the top-`k` predictions canonically equals `gold`, else 0.

    `predictions` are reactant-set SMILES strings (may contain `.`-separated
    reactants). `gold` is the gold reactant set in the same format.
    """
    gold_canon = canon_rxn_side(gold, policy)
    if gold_canon is None:
        return 0
    for pred in predictions[:k]:
        pred_canon = canon_rxn_side(pred, policy) if pred is not None else None
        if pred_canon is not None and pred_canon == gold_canon:
            return 1
    return 0


def invalid_smiles_rate(predictions: Iterable[str]) -> float:
    """Fraction of predicted SMILES strings that fail to parse with RDKit.

    Each prediction may be a `.`-joined reactant set; we count it invalid if
    ANY fragment fails to parse.
    """
    total = 0
    invalid = 0
    for p in predictions:
        total += 1
        if p is None:
            invalid += 1
            continue
        parts = [x.strip() for x in p.split(".") if x.strip()]
        if not parts or any(not is_valid_smiles(x) for x in parts):
            invalid += 1
    return invalid / total if total else 0.0


def _largest_fragment_canon(
    smiles: str, policy: CanonPolicy = DEFAULT_POLICY
) -> Optional[str]:
    if smiles is None:
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    frags = Chem.GetMolFrags(m, asMols=True)
    if not frags:
        return None
    biggest = max(frags, key=lambda x: x.GetNumHeavyAtoms())
    if policy.remove_atom_mapping:
        for atom in biggest.GetAtoms():
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(
        biggest,
        isomericSmiles=policy.isomeric_smiles,
        kekuleSmiles=policy.kekulize,
    )


def maxfrag_accuracy(
    predictions: list[str],
    gold: str,
    k: int,
    policy: CanonPolicy = DEFAULT_POLICY,
) -> int:
    """1 if the largest fragment of any top-`k` prediction equals the largest
    fragment of `gold`, else 0.

    Used in some retrosynthesis papers (e.g., Molecular Transformer) as a more
    forgiving variant of exact match — credits a prediction that gets the main
    backbone right even if a small fragment differs.
    """
    gold_largest = _largest_fragment_canon(gold, policy)
    if gold_largest is None:
        return 0
    for pred in predictions[:k]:
        pl = _largest_fragment_canon(pred, policy)
        if pl is not None and pl == gold_largest:
            return 1
    return 0


def candidate_diversity(
    predictions: list[str], policy: CanonPolicy = DEFAULT_POLICY
) -> float:
    """Fraction of unique canonical predictions in the top-k list.

    Returns 1.0 if all valid predictions are distinct molecules, near 0 if all
    identical. Cheap, model-agnostic diversity proxy.
    """
    canons: list[str] = []
    for p in predictions:
        if p is None:
            continue
        c = canon_rxn_side(p, policy)
        if c is not None:
            canons.append(c)
    if not canons:
        return 0.0
    return len(set(canons)) / len(canons)


def round_trip_accuracy(
    predictions: list[str],
    product: str,
    k: int,
    forward_fn: Callable[[str], list[str]],
    policy: CanonPolicy = DEFAULT_POLICY,
) -> int:
    """1 if running `forward_fn` on any top-`k` predicted reactants reproduces
    `product`, else 0.

    `forward_fn(reactants)` -> list[predicted_product_smiles]. Round-trip
    succeeds if `product` canonically equals any predicted product from any
    top-`k` retrosynthesis. Forward-model exceptions are caught and treated as
    a miss for that candidate (round-trip is a noisy oracle by design).
    """
    product_canon = canon(product, policy)
    if product_canon is None:
        return 0
    for pred in predictions[:k]:
        if pred is None or canon_rxn_side(pred, policy) is None:
            continue
        try:
            forward_predictions = forward_fn(pred)
        except Exception:
            continue
        for fp in forward_predictions:
            if canon(fp, policy) == product_canon:
                return 1
    return 0
