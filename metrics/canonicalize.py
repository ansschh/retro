"""Canonical SMILES function used everywhere in the harness.

Every exact-match comparison goes through `canon` so that two SMILES strings
represent the same molecule iff `canon(a) == canon(b)` under the same policy.
Policy is loaded from `configs/protocol.yaml` (the `canonicalization` section).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class CanonPolicy:
    isomeric_smiles: bool = True
    kekulize: bool = False
    remove_atom_mapping: bool = True
    canonical_tautomer: bool = False
    charge_handling: str = "preserve"  # preserve | neutralize
    salt_handling: str = "split_largest_fragment"  # split_largest_fragment | keep_all
    invalid_smiles_policy: str = "count_then_skip_in_topk"

    @classmethod
    def from_protocol(cls, protocol: dict) -> "CanonPolicy":
        c = protocol.get("canonicalization", {})
        return cls(
            isomeric_smiles=c.get("isomeric_smiles", True),
            kekulize=c.get("kekulize", False),
            remove_atom_mapping=c.get("remove_atom_mapping", True),
            canonical_tautomer=c.get("canonical_tautomer", False),
            charge_handling=c.get("charge_handling", "preserve"),
            salt_handling=c.get("salt_handling", "split_largest_fragment"),
            invalid_smiles_policy=c.get(
                "invalid_smiles_policy", "count_then_skip_in_topk"
            ),
        )


DEFAULT_POLICY = CanonPolicy()


def canon(smiles: str, policy: CanonPolicy = DEFAULT_POLICY) -> Optional[str]:
    """Return the canonical SMILES for `smiles` under `policy`, or None if invalid.

    Single-molecule canonicalization. Applies `salt_handling` (largest-fragment
    selection by default) — for reactant sets, use `canon_rxn_side` instead.
    """
    if smiles is None or not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    if policy.remove_atom_mapping:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
    if policy.salt_handling == "split_largest_fragment":
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    elif policy.salt_handling != "keep_all":
        raise ValueError(f"unknown salt_handling: {policy.salt_handling}")
    if policy.charge_handling == "neutralize":
        try:
            from rdkit.Chem.MolStandardize import rdMolStandardize

            mol = rdMolStandardize.Uncharger().uncharge(mol)
        except ImportError:
            pass
    elif policy.charge_handling != "preserve":
        raise ValueError(f"unknown charge_handling: {policy.charge_handling}")
    try:
        return Chem.MolToSmiles(
            mol,
            isomericSmiles=policy.isomeric_smiles,
            kekuleSmiles=policy.kekulize,
        )
    except Exception:
        return None


def canon_rxn_side(
    smiles: str, policy: CanonPolicy = DEFAULT_POLICY
) -> Optional[str]:
    """Canonicalize one side of a reaction (reactants or products).

    The input may be `.`-separated. Each fragment is parsed and canonicalized
    independently; fragments are then sorted canonically and rejoined. Salt
    handling does NOT apply here: every reactant in a multi-reactant
    retrosynthesis is meaningful, so we preserve all fragments.

    Returns None if any fragment fails to parse.
    """
    if smiles is None or not isinstance(smiles, str):
        return None
    parts = [p.strip() for p in smiles.split(".") if p.strip()]
    canons: list[str] = []
    for p in parts:
        m = Chem.MolFromSmiles(p)
        if m is None or m.GetNumAtoms() == 0:
            return None
        if policy.remove_atom_mapping:
            for atom in m.GetAtoms():
                atom.SetAtomMapNum(0)
        try:
            canons.append(
                Chem.MolToSmiles(
                    m,
                    isomericSmiles=policy.isomeric_smiles,
                    kekuleSmiles=policy.kekulize,
                )
            )
        except Exception:
            return None
    if not canons:
        return None
    return ".".join(sorted(canons))


def is_valid_smiles(smiles: str) -> bool:
    """True if RDKit can parse the SMILES into at least one atom, False otherwise.

    Empty strings are treated as invalid even though RDKit returns an empty mol.
    """
    if smiles is None or not isinstance(smiles, str) or not smiles.strip():
        return False
    m = Chem.MolFromSmiles(smiles)
    return m is not None and m.GetNumAtoms() > 0
