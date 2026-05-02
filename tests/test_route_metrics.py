"""Multistep route-metric tests."""
from metrics.route_metrics import (
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


def make_simple_route() -> RouteNode:
    # target = CCC, made from CC + C
    return RouteNode(
        smiles="CCC",
        children=[
            RouteNode(smiles="CC", in_stock=True),
            RouteNode(smiles="C", in_stock=True),
        ],
    )


def test_routes_match_identical():
    assert routes_match(make_simple_route(), make_simple_route())


def test_routes_match_with_permuted_children():
    a = RouteNode(
        smiles="CCC",
        children=[RouteNode(smiles="CC"), RouteNode(smiles="C")],
    )
    b = RouteNode(
        smiles="CCC",
        children=[RouteNode(smiles="C"), RouteNode(smiles="CC")],
    )
    assert routes_match(a, b)


def test_routes_dont_match_different_target():
    a = RouteNode(smiles="CCC", children=[RouteNode(smiles="CC")])
    b = RouteNode(smiles="CCN", children=[RouteNode(smiles="CC")])
    assert not routes_match(a, b)


def test_routes_dont_match_different_structure():
    a = RouteNode(smiles="CCC", children=[RouteNode(smiles="CC")])
    b = RouteNode(
        smiles="CCC",
        children=[RouteNode(smiles="CC", children=[RouteNode(smiles="C")])],
    )
    assert not routes_match(a, b)


def test_route_length():
    r = RouteNode(
        smiles="CCC",
        children=[
            RouteNode(
                smiles="CC", children=[RouteNode(smiles="C")]
            ),
            RouteNode(smiles="N"),
        ],
    )
    # Two internal nodes
    assert route_length(r) == 2


def test_solve_rate_in_stock():
    assert solve_rate([make_simple_route()], stock={"CC", "C"}) == 1


def test_solve_rate_not_in_stock():
    assert solve_rate([make_simple_route()], stock={"CN"}) == 0


def test_stock_coverage_partial():
    cov = stock_coverage(make_simple_route(), stock={"CC"})
    assert abs(cov - 0.5) < 1e-9


def test_top_k_route_accuracy():
    pred1 = RouteNode(smiles="CCC", children=[RouteNode(smiles="CC")])
    pred2 = make_simple_route()
    gold = make_simple_route()
    assert top_k_route_accuracy([pred1, pred2], [gold], k=2) == 1
    assert top_k_route_accuracy([pred1, pred2], [gold], k=1) == 0


def test_route_edit_distance_identical():
    assert route_edit_distance(make_simple_route(), make_simple_route()) == 0


def test_route_edit_distance_one_label_change():
    a = RouteNode(smiles="CCC", children=[RouteNode(smiles="CC")])
    b = RouteNode(smiles="CCN", children=[RouteNode(smiles="CC")])
    assert route_edit_distance(a, b) == 1


def test_route_diversity_unique():
    r1 = RouteNode(smiles="CCC", children=[RouteNode(smiles="CC")])
    r2 = RouteNode(smiles="CCN", children=[RouteNode(smiles="CC")])
    assert route_diversity([r1, r2]) == 1.0


def test_route_diversity_repeated():
    r = make_simple_route()
    assert abs(route_diversity([r, r, r]) - 1 / 3) < 1e-9


def test_route_diversity_empty():
    assert route_diversity([]) == 0.0


def test_from_paroutes_json_simple():
    d = {
        "smiles": "CCC",
        "type": "mol",
        "in_stock": False,
        "children": [
            {
                "type": "reaction",
                "children": [
                    {
                        "smiles": "CC",
                        "type": "mol",
                        "in_stock": True,
                        "children": [],
                    },
                    {
                        "smiles": "C",
                        "type": "mol",
                        "in_stock": True,
                        "children": [],
                    },
                ],
            }
        ],
    }
    r = from_paroutes_json(d)
    assert r.smiles == "CCC"
    assert {c.smiles for c in r.children} == {"CC", "C"}
    assert all(c.in_stock for c in r.children)
