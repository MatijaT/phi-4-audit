"""Parser for the Panzer-Schnetz `Periods` ancillary file (arXiv:1603.04289, "The Galois
coaction on phi^4 periods"). This file is the only external data this repository uses.

Each entry has the form
    Period[l,m]:=[edge_list, motivic_period, period, numeric, c2, ancestor, aut]:
where edge_list is the completed (4-regular) graph, c2 is the c2 invariant annotation
("-1", "0", "-z[2]", "-modularform[..]", "-Fpsequence[..]", ...), ancestor is the
ancestor pointer, and the last field is |Aut| of the completed graph. The file lists
every completed primitive phi^4 graph through 11 loops EXCEPT products (graphs with a
3-vertex cut); those are regenerated in census.py.
"""
import hashlib
import os
import re
import urllib.request

import networkx as nx

PERIODS_URL = "https://arxiv.org/src/1603.04289v2/anc/Periods"
PERIODS_MD5 = "a45427c75d6aec6e3c07e95efc74c0d5"

_ENTRY = re.compile(r"Period\[(\d+),\s*(\d+)\]\s*:=\s*\[(.*)\]\s*:\s*$")


def fetch(path):
    """Download the Periods file to `path` (if absent) and verify its checksum."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        print(f"downloading {PERIODS_URL} -> {path}")
        urllib.request.urlretrieve(PERIODS_URL, path)
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    if md5 != PERIODS_MD5:
        raise RuntimeError(f"{path}: md5 {md5} != expected {PERIODS_MD5}")
    return path


def _split_top(s):
    """Split on commas at bracket depth 0."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [x.strip() for x in out]


def parse(path):
    """Return {(l, m): fields}."""
    entries = {}
    with open(path) as fh:
        for line in fh:
            m = _ENTRY.match(line.strip())
            if m:
                entries[(int(m.group(1)), int(m.group(2)))] = _split_top(m.group(3))
    return entries


def to_graph(edge_field):
    """'[{1, 2}, {1, 3}, ...]' -> simple networkx Graph on 0..n-1."""
    pairs = [(int(a), int(b)) for a, b in re.findall(r"\{\s*(\d+)\s*,\s*(\d+)\s*\}", edge_field)]
    verts = sorted({v for e in pairs for v in e})
    idx = {v: i for i, v in enumerate(verts)}
    G = nx.Graph()
    G.add_nodes_from(range(len(verts)))
    G.add_edges_from((idx[a], idx[b]) for a, b in pairs)
    return G


def c2_label(c2_field):
    """1 if the annotated c2 invariant is 0 (weight-drop proxy), 0 if nonzero, None if absent."""
    s = c2_field.strip()
    if s == "0":
        return 1
    if s == "FAIL" or s.startswith("c2("):
        return None
    return 0
