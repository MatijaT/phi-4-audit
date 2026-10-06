"""Checks on the built datasets (run `make data` first; skipped otherwise)."""
import csv
import os
from collections import Counter

import networkx as nx
import pytest

from phi4audit import graphs
from phi4audit.c2_bruteforce import c2

CENSUS = "data/built/census_L3-9.csv"
pytestmark = pytest.mark.skipif(not os.path.exists(CENSUS), reason="run `make data` first")


@pytest.fixture(scope="module")
def rows():
    return list(csv.DictReader(open(CENSUS)))


def test_census_counts(rows):
    # completed primitive phi^4 graphs per loop order (Schnetz's census, arXiv:0801.2856)
    assert Counter(int(r["L"]) for r in rows) == {3: 1, 4: 1, 5: 2, 6: 5, 7: 14, 8: 49, 9: 227}


def test_three_vertex_cut_forces_label(rows):
    vc3 = [r for r in rows if r["vertex_conn"] == "3"]
    assert len(vc3) == 50 and all(r["weight_drop"] == "1" for r in vc3)
    assert all(r["source"] == "product" for r in vc3)


def test_four_positive_families(rows):
    pos = [r for r in rows if r["ancestor_type"] == "prime" and r["weight_drop"] == "1"]
    assert len(pos) == 15 and len({r["family"] for r in pos}) == 4


def test_ancestor_confluence(rows):
    for r in rows[::7]:
        G = nx.from_graph6_bytes(r["g6"].encode())
        assert len({graphs.ancestor(G, s) for s in range(3)}) == 1


def test_labels_against_bruteforce(rows):
    # independent spot check of the Periods annotation at p = 2, 3 on small graphs
    for r in rows:
        if int(r["L"]) <= 6:
            G = nx.from_graph6_bytes(r["g6"].encode())
            zero = all(c2(G, p) == 0 for p in (2, 3))
            assert zero == (r["weight_drop"] == "1"), r["g6"]
