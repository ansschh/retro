"""Single-step metric tests."""
from metrics.single_step_metrics import (
    candidate_diversity,
    invalid_smiles_rate,
    maxfrag_accuracy,
    round_trip_accuracy,
    top_k_exact_match,
)


def test_top_k_match_top_1():
    assert top_k_exact_match(["CCO.CC(=O)O"], "CCO.CC(=O)O", k=1) == 1


def test_top_k_match_with_permuted_reactants():
    assert top_k_exact_match(["CC(=O)O.CCO"], "CCO.CC(=O)O", k=1) == 1


def test_top_k_no_match():
    assert top_k_exact_match(["CCC"], "CCO", k=1) == 0


def test_top_k_match_at_position_3():
    preds = ["CCC", "CCN", "CCO", "CCS"]
    assert top_k_exact_match(preds, "CCO", k=3) == 1
    assert top_k_exact_match(preds, "CCO", k=2) == 0


def test_top_k_handles_invalid_predictions():
    preds = ["not_smiles", "CCO"]
    assert top_k_exact_match(preds, "CCO", k=2) == 1


def test_top_k_handles_none_in_predictions():
    preds = [None, "CCO"]
    assert top_k_exact_match(preds, "CCO", k=2) == 1


def test_invalid_smiles_rate():
    preds = ["CCO", "CCN", "not_smiles", "CC(", "C[C@H](N)C(=O)O"]
    assert abs(invalid_smiles_rate(preds) - 2 / 5) < 1e-9


def test_invalid_smiles_rate_with_dotted_sets():
    # If any fragment is invalid, the whole prediction is invalid
    preds = ["CCO.CC(=O)O", "CCO.not_smiles"]
    assert abs(invalid_smiles_rate(preds) - 0.5) < 1e-9


def test_invalid_smiles_rate_handles_none():
    assert abs(invalid_smiles_rate([None, "CCO"]) - 0.5) < 1e-9


def test_maxfrag_accuracy_matches_largest():
    # Right backbone, different small fragment — counts under maxfrag
    pred = "c1ccc(C(=O)O)cc1.O"
    gold = "c1ccc(C(=O)O)cc1.[Na+]"
    assert maxfrag_accuracy([pred], gold, k=1) == 1


def test_maxfrag_accuracy_misses_when_largest_differs():
    pred = "c1ccccc1C.O"
    gold = "c1ccc(C(=O)O)cc1.O"
    assert maxfrag_accuracy([pred], gold, k=1) == 0


def test_candidate_diversity_all_unique():
    preds = ["CCO", "CCN", "CCC"]
    assert candidate_diversity(preds) == 1.0


def test_candidate_diversity_collapses_synonyms():
    # OCC and CCO are the same molecule — treated as one
    preds = ["CCO", "OCC", "CCN"]
    assert abs(candidate_diversity(preds) - 2 / 3) < 1e-9


def test_candidate_diversity_empty():
    assert candidate_diversity([]) == 0.0


def test_round_trip_accuracy_passes_when_forward_recovers_product():
    def fake_forward(_):
        return ["CCO"]

    assert (
        round_trip_accuracy(
            ["CC(=O)O.[H]"], "CCO", k=1, forward_fn=fake_forward
        )
        == 1
    )


def test_round_trip_accuracy_fails_when_forward_returns_wrong():
    def fake_forward(_):
        return ["CCN"]

    assert (
        round_trip_accuracy(
            ["CC(=O)O.[H]"], "CCO", k=1, forward_fn=fake_forward
        )
        == 0
    )


def test_round_trip_accuracy_handles_forward_exception():
    def broken_forward(_):
        raise RuntimeError("forward model crashed")

    # Should NOT propagate; returns 0 (the candidate counts as a miss)
    assert (
        round_trip_accuracy(
            ["CCO"], "CCO", k=1, forward_fn=broken_forward
        )
        == 0
    )
