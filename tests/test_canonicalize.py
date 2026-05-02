"""Canonical-SMILES tests — gotchas first.

Verify canon(a) == canon(b) iff a and b represent the same molecule under the
protocol's canonicalization policy.
"""
from metrics.canonicalize import (
    CanonPolicy,
    canon,
    canon_rxn_side,
    is_valid_smiles,
)


def test_basic_canonicalization():
    assert canon("CCO") == canon("OCC")
    assert canon("c1ccccc1") == canon("C1=CC=CC=C1")


def test_invalid_smiles_returns_none():
    assert canon("not a smiles") is None
    assert canon("") is None
    assert canon(None) is None
    assert canon("C(") is None


def test_atom_mapping_removed_by_default():
    assert canon("[CH3:1][CH2:2][OH:3]") == canon("CCO")


def test_atom_mapping_kept_when_policy_says_so():
    p = CanonPolicy(remove_atom_mapping=False)
    assert canon("[CH3:1][CH2:2][OH:3]", p) != canon("CCO", p)


def test_salt_split_largest_fragment():
    # Sodium acetate -> the largest fragment is acetate
    assert canon("CC(=O)[O-].[Na+]") == canon("CC(=O)[O-]")


def test_salt_keep_all_preserves_fragments():
    p = CanonPolicy(salt_handling="keep_all")
    a = canon("CC(=O)[O-].[Na+]", p)
    assert a is not None and "." in a


def test_charge_preserve_default():
    # Acetate (charged) and acetic acid (neutral) are different under preserve
    assert canon("CC(=O)[O-]") != canon("CC(=O)O")


def test_isomeric_default():
    # Two stereoisomers should differ
    assert canon("C[C@H](N)C(=O)O") != canon("C[C@@H](N)C(=O)O")


def test_canon_rxn_side_orders_reactants():
    assert canon_rxn_side("CCO.CC(=O)O") == canon_rxn_side("CC(=O)O.CCO")


def test_canon_rxn_side_invalid_returns_none():
    assert canon_rxn_side("CCO.not_smiles") is None


def test_canon_rxn_side_empty_fragments_skipped():
    # Trailing dot must not produce an empty parse failure
    assert canon_rxn_side("CCO.CCN.") == canon_rxn_side("CCO.CCN")


def test_canon_rxn_side_keeps_all_reactants():
    # Salt handling does NOT apply to rxn sides; both reactants must persist
    a = canon_rxn_side("CCO.CC(=O)O")
    assert a is not None and "." in a


def test_is_valid_smiles():
    assert is_valid_smiles("CCO")
    assert not is_valid_smiles("not_smiles")
    assert not is_valid_smiles("")
    assert not is_valid_smiles(None)
