import itertools

import networkx as nx

from phi4audit import graphs
from phi4audit.c2_bruteforce import c2
from phi4audit.features import features

K5 = nx.complete_graph(5)


def k5_squared():
    return graphs.glue(K5, (0, 1, 2), K5, (0, 1, 2))


def test_k5_is_primitive_and_irreducible():
    assert graphs.is_primitive(K5)
    assert not graphs.dt_moves(K5) and not graphs.split_moves(K5)
    assert graphs.ancestor(K5) == (graphs.K5,)


def test_product_of_two_k5():
    P = k5_squared()                       # the unique product at 5 loops
    assert P.number_of_nodes() == 7 and graphs.is_4regular(P) and graphs.is_primitive(P)
    assert nx.node_connectivity(P) == 3
    for seed in range(3):                  # reduction is independent of move order
        assert graphs.ancestor(P, seed) == (graphs.K5, graphs.K5)


def test_c2_bruteforce_anchors():
    # c2(K5) = -1 at every prime; products have c2 = 0
    for p in (2, 3, 5):
        assert c2(K5, p) == p - 1
    for p in (2, 3):
        assert c2(k5_squared(), p) == 0


def test_features_are_isomorphism_invariant():
    P = k5_squared()
    perm = dict(zip(P.nodes(), reversed(list(P.nodes()))))
    assert features(P) == features(nx.relabel_nodes(P, perm))
