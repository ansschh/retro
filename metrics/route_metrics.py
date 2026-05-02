"""Multistep route metrics.

A route is a tree of reactions. Each node carries a SMILES; internal nodes
(reactions) have children that are their reactants; leaf nodes are stock
molecules. Format mirrors PaRoutes' JSON schema for compatibility.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .canonicalize import DEFAULT_POLICY, CanonPolicy, canon


@dataclass
class RouteNode:
    """A node in a retrosynthesis route tree."""

    smiles: str
    children: list["RouteNode"] = field(default_factory=list)
    in_stock: bool = False

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def all_leaves(self) -> list["RouteNode"]:
        if self.is_leaf:
            return [self]
        out: list[RouteNode] = []
        for c in self.children:
            out.extend(c.all_leaves())
        return out

    def n_reactions(self) -> int:
        """Number of internal nodes in this route (i.e., reaction steps)."""
        if self.is_leaf:
            return 0
        return 1 + sum(c.n_reactions() for c in self.children)


def _canonicalize_tree(
    node: RouteNode, policy: CanonPolicy = DEFAULT_POLICY
) -> Optional[RouteNode]:
    cs = canon(node.smiles, policy)
    if cs is None:
        return None
    new_children: list[RouteNode] = []
    for c in node.children:
        nc = _canonicalize_tree(c, policy)
        if nc is None:
            return None
        new_children.append(nc)
    new_children.sort(key=lambda x: x.smiles)
    return RouteNode(smiles=cs, children=new_children, in_stock=node.in_stock)


def _trees_equal(a: RouteNode, b: RouteNode) -> bool:
    if a.smiles != b.smiles or len(a.children) != len(b.children):
        return False
    return all(_trees_equal(ac, bc) for ac, bc in zip(a.children, b.children))


def routes_match(
    predicted: RouteNode,
    gold: RouteNode,
    policy: CanonPolicy = DEFAULT_POLICY,
) -> bool:
    """True if two routes are structurally and chemically identical (after
    canonicalization). Order of reactants does not matter — children are sorted
    by canonical SMILES inside `_canonicalize_tree`."""
    pc = _canonicalize_tree(predicted, policy)
    gc = _canonicalize_tree(gold, policy)
    if pc is None or gc is None:
        return False
    return _trees_equal(pc, gc)


def top_k_route_accuracy(
    predicted_routes: list[RouteNode],
    gold_routes: list[RouteNode],
    k: int,
    policy: CanonPolicy = DEFAULT_POLICY,
) -> int:
    """1 if any of top-`k` predicted routes matches any gold route exactly.

    PaRoutes provides multiple gold routes per target; matching ANY counts.
    """
    for pred in predicted_routes[:k]:
        for gold in gold_routes:
            if routes_match(pred, gold, policy):
                return 1
    return 0


def solve_rate(
    predicted_routes: list[RouteNode],
    stock: set[str],
    policy: CanonPolicy = DEFAULT_POLICY,
) -> int:
    """1 if any predicted route's leaves are all canonically in `stock`, else 0."""
    canon_stock = {c for c in (canon(s, policy) for s in stock) if c is not None}
    for r in predicted_routes:
        leaves_canon = {canon(l.smiles, policy) for l in r.all_leaves()}
        if leaves_canon and leaves_canon.issubset(canon_stock):
            return 1
    return 0


def stock_coverage(
    route: RouteNode,
    stock: set[str],
    policy: CanonPolicy = DEFAULT_POLICY,
) -> float:
    """Fraction of leaves of `route` that are canonically in `stock`."""
    canon_stock = {c for c in (canon(s, policy) for s in stock) if c is not None}
    leaves = route.all_leaves()
    if not leaves:
        return 0.0
    n_in = sum(1 for l in leaves if canon(l.smiles, policy) in canon_stock)
    return n_in / len(leaves)


def route_length(route: RouteNode) -> int:
    """Number of reaction steps in the route."""
    return route.n_reactions()


def _subtree_size(node: RouteNode) -> int:
    return 1 + sum(_subtree_size(c) for c in node.children)


def _tree_edit(a: RouteNode, b: RouteNode) -> int:
    """Approximate tree edit distance with greedy child matching.

    Cost 1 for any insert/delete/relabel. Adequate for our small route trees
    (typical depth <= 8). Not full Zhang-Shasha; for routes this size the
    approximation matches in practice.
    """
    relabel = 0 if a.smiles == b.smiles else 1
    if not a.children and not b.children:
        return relabel
    if not a.children:
        return _subtree_size(b) - (1 if a.smiles == b.smiles else 0)
    if not b.children:
        return _subtree_size(a) - (1 if a.smiles == b.smiles else 0)
    children_cost = 0
    used_b = [False] * len(b.children)
    for ac in a.children:
        best = -1
        best_j = -1
        for j, bc in enumerate(b.children):
            if used_b[j]:
                continue
            d = _tree_edit(ac, bc)
            if best_j < 0 or d < best:
                best = d
                best_j = j
        if best_j >= 0:
            used_b[best_j] = True
            children_cost += best
        else:
            children_cost += _subtree_size(ac)
    for j, bc in enumerate(b.children):
        if not used_b[j]:
            children_cost += _subtree_size(bc)
    return relabel + children_cost


def route_edit_distance(
    a: RouteNode, b: RouteNode, policy: CanonPolicy = DEFAULT_POLICY
) -> int:
    """Approximate tree edit distance between canonicalized routes.

    Returns -1 if either route contains an invalid SMILES.
    """
    ca = _canonicalize_tree(a, policy)
    cb = _canonicalize_tree(b, policy)
    if ca is None or cb is None:
        return -1
    return _tree_edit(ca, cb)


def _tree_signature(node: RouteNode) -> tuple:
    return (node.smiles, tuple(_tree_signature(c) for c in node.children))


def route_diversity(
    routes: list[RouteNode], policy: CanonPolicy = DEFAULT_POLICY
) -> float:
    """Fraction of routes that are unique under canonical-tree equality.

    Returns len(unique_signatures) / len(routes). 1.0 means all distinct.
    """
    if not routes:
        return 0.0
    sigs: list[tuple] = []
    for r in routes:
        cr = _canonicalize_tree(r, policy)
        if cr is not None:
            sigs.append(_tree_signature(cr))
    if not sigs:
        return 0.0
    return len(set(sigs)) / len(sigs)


def from_paroutes_json(d: dict) -> RouteNode:
    """Convert a PaRoutes-format JSON dict into a RouteNode tree.

    PaRoutes nodes carry `smiles`, `type` (`mol` or `reaction`), `in_stock`,
    `children`. Mol nodes have reaction children; reaction children have mol
    children. Our `RouteNode` collapses reaction nodes — a mol's children are
    the precursor mols of its reaction.
    """
    if d.get("type", "mol") == "mol":
        children: list[RouteNode] = []
        for c in d.get("children", []):
            for grandchild in c.get("children", []):
                children.append(from_paroutes_json(grandchild))
        return RouteNode(
            smiles=d["smiles"],
            children=children,
            in_stock=d.get("in_stock", False),
        )
    # Reaction node at top level — treat as wrapper.
    if d.get("children"):
        return from_paroutes_json(d["children"][0])
    raise ValueError("malformed PaRoutes node")
