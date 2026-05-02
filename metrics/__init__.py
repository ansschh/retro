"""Reproducible single-step and multistep retrosynthesis metrics.

Every comparison routes through `metrics.canonicalize.canon` so exact-match
checks obey the policy in `configs/protocol.yaml`. Two SMILES strings represent
the same molecule iff their canonical forms are equal under the same policy.
"""
from .canonicalize import (
    CanonPolicy,
    DEFAULT_POLICY,
    canon,
    canon_rxn_side,
    is_valid_smiles,
)
from .single_step_metrics import (
    candidate_diversity,
    invalid_smiles_rate,
    maxfrag_accuracy,
    round_trip_accuracy,
    top_k_exact_match,
)
from .route_metrics import (
    RouteNode,
    from_paroutes_json,
    route_diversity,
    route_edit_distance,
    route_length,
    routes_match,
    solve_rate,
    stock_coverage,
    top_k_route_accuracy,
)
from .protocol import load_protocol, protocol_sha

__all__ = [
    "CanonPolicy",
    "DEFAULT_POLICY",
    "RouteNode",
    "candidate_diversity",
    "canon",
    "canon_rxn_side",
    "from_paroutes_json",
    "invalid_smiles_rate",
    "is_valid_smiles",
    "load_protocol",
    "maxfrag_accuracy",
    "protocol_sha",
    "round_trip_accuracy",
    "route_diversity",
    "route_edit_distance",
    "route_length",
    "routes_match",
    "solve_rate",
    "stock_coverage",
    "top_k_exact_match",
    "top_k_route_accuracy",
]
