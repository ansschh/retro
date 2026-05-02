"""Protocol loader tests."""
from metrics.protocol import load_protocol, protocol_sha


def test_load_protocol_has_required_sections():
    d = load_protocol()
    required = {
        "single_step",
        "multistep",
        "stock_lists",
        "canonicalization",
        "single_step_defaults",
        "multistep_defaults",
        "reporting",
    }
    assert required.issubset(d.keys())


def test_canonicalization_section_has_expected_keys():
    d = load_protocol()
    c = d["canonicalization"]
    expected = {
        "rdkit_version",
        "isomeric_smiles",
        "kekulize",
        "remove_atom_mapping",
        "canonical_tautomer",
        "charge_handling",
        "salt_handling",
        "invalid_smiles_policy",
    }
    assert expected.issubset(c.keys())


def test_protocol_sha_stable_and_full_length():
    s1 = protocol_sha()
    s2 = protocol_sha()
    assert s1 == s2
    assert len(s1) == 64
