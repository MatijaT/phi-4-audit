"""Completed primitive phi^4 graphs: canonical forms, primitivity, products, and the
ancestor reduction (double-triangle reductions + product splits, HSSY arXiv:1812.08751 §3).

All graphs are 4-regular *completions* (the c2 invariant is completion-invariant).
"""
import itertools
import random

import igraph
import networkx as nx


# --------------------------------------------------------------------- canonical forms
def canon(G):
    """Canonical form (BLISS via igraph): (n, sorted edge tuple)."""
    nodes = sorted(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    g = igraph.Graph(n=len(nodes), edges=[(idx[u], idx[v]) for u, v in G.edges()])
    gc = g.permute_vertices(g.canonical_permutation())
    return (len(nodes), tuple(sorted(tuple(sorted(e)) for e in gc.get_edgelist())))


def canon_graph(key):
    n, edges = key
    H = nx.Graph()
    H.add_nodes_from(range(n))
    H.add_edges_from(edges)
    return H


def canon_g6(key):
    return nx.to_graph6_bytes(canon_graph(key), header=False).decode().strip()


# --------------------------------------------------------------------- primitivity
def is_4regular(G):
    return all(d == 4 for _, d in G.degree())


def is_primitive(G):
    """4-regular, connected, and no edge cut of size <= 4 with >= 2 vertices on each side
    (only the trivial single-vertex 4-cuts are allowed)."""
    if not is_4regular(G) or not nx.is_connected(G):
        return False
    adj = {u: set(G.neighbors(u)) for u in G.nodes()}
    V = list(G.nodes())
    for k in range(2, len(V) // 2 + 1):
        for S in itertools.combinations(V, k):
            Sset = set(S)
            if 4 * k - sum(len(adj[u] & Sset) for u in S) <= 4:   # boundary = 4k - 2 e(S)
                return False
    return True


def loop_order(G):
    return G.number_of_nodes() - 2


# --------------------------------------------------------------------- products
def triangles(G):
    return [t for t in itertools.combinations(sorted(G.nodes()), 3)
            if G.has_edge(t[0], t[1]) and G.has_edge(t[1], t[2]) and G.has_edge(t[0], t[2])]


def glue(G1, t1, G2, t2):
    """Product of two completed graphs along triangles t1 in G1 and t2 in G2: delete both
    triangles' edges and identify t2[i] with t1[i]. Loop order L1 + L2 - 1."""
    off = max(G1.nodes()) + 1
    ident = {t2[i]: t1[i] for i in range(3)}
    tri1 = {frozenset(e) for e in itertools.combinations(t1, 2)}
    tri2 = {frozenset(e) for e in itertools.combinations(t2, 2)}
    H = nx.Graph()
    H.add_edges_from(e for e in G1.edges() if frozenset(e) not in tri1)
    for u, v in G2.edges():
        if frozenset((u, v)) in tri2:
            continue
        H.add_edge(ident.get(u, u + off), ident.get(v, v + off))
    return nx.convert_node_labels_to_integers(H)


def all_products(by_loop, L):
    """All products at loop order L, given the complete lists of completed primitive graphs
    at every lower loop order (products included, so iterated products are generated)."""
    out = {}
    for L1 in range(3, L - 2 + 1):
        L2 = L + 1 - L1
        if L2 < L1:
            break
        for G1 in by_loop.get(L1, []):
            T1 = triangles(G1)
            for G2 in by_loop.get(L2, []):
                for t1 in T1:
                    for t2 in triangles(G2):
                        for perm in itertools.permutations(t2):
                            H = glue(G1, t1, G2, perm)
                            if H.number_of_edges() != 2 * H.number_of_nodes():
                                continue
                            out.setdefault(canon(H), H)
    return out


# --------------------------------------------------------------------- ancestor reduction
def dt_moves(G):
    """Double-triangle reductions: edge (A,B) in exactly two triangles ABC, ABD; delete B
    (neighbours A,C,D,E), add edges CD and AE. Skipped if a multi-edge would be created."""
    moves = []
    adj = {u: set(G.neighbors(u)) for u in G.nodes()}
    for a, b in G.edges():
        common = adj[a] & adj[b]
        if len(common) != 2:
            continue
        C, D = tuple(common)
        if G.has_edge(C, D):
            continue
        for A, B in ((a, b), (b, a)):
            rest = adj[B] - {A, C, D}
            if len(rest) != 1:
                continue
            E = next(iter(rest))
            if E in (A, C, D) or G.has_edge(A, E):
                continue
            moves.append((B, A, C, D, E))
    return moves


def apply_dt(G, move):
    B, A, C, D, E = move
    H = G.copy()
    H.remove_node(B)
    H.add_edge(C, D)
    H.add_edge(A, E)
    return H


def _piece(G, S, cut):
    H = G.subgraph(set(S) | set(cut)).copy()
    x, y, z = cut
    H.add_edges_from([(x, y), (y, z), (x, z)])
    return H if is_4regular(H) else None


def split_moves(G):
    """3-vertex cuts {x,y,z} splitting G into two completed primitive pieces."""
    moves = []
    for cut in itertools.combinations(list(G.nodes()), 3):
        H = G.copy()
        H.remove_nodes_from(cut)
        comps = [set(c) for c in nx.connected_components(H)]
        if len(comps) < 2:
            continue
        for i in range(len(comps)):
            S1 = comps[i]
            S2 = set().union(*[comps[j] for j in range(len(comps)) if j != i])
            g1, g2 = _piece(G, S1, cut), _piece(G, S2, cut)
            if g1 is not None and g2 is not None and is_primitive(g1) and is_primitive(g2):
                moves.append((cut, frozenset(S1)))
                break
    return moves


def apply_split(G, move):
    cut, S1 = move
    H = G.copy()
    H.remove_nodes_from(cut)
    S2 = set().union(*[c for c in nx.connected_components(H) if not (c & S1)])
    return _piece(G, S1, cut), _piece(G, S2, cut)


def ancestor(G0, seed=0):
    """Reduce by DT moves and product splits (random order) to the multiset of irreducible
    pieces; returns a sorted tuple of canonical forms. The result is independent of the
    order of moves (a theorem; checked in tests/ with several seeds)."""
    rng = random.Random(seed)
    work, irreducible = [G0], []
    while work:
        H = work.pop(rng.randrange(len(work)))
        moves = [("dt", m) for m in dt_moves(H)] + [("sp", m) for m in split_moves(H)]
        if not moves:
            irreducible.append(canon(H))
            continue
        kind, m = rng.choice(moves)
        if kind == "dt":
            H2 = apply_dt(H, m)
            assert is_primitive(H2), f"DT move broke primitivity: {m}"
            work.append(H2)
        else:
            g1, g2 = apply_split(H, m)
            assert is_primitive(g1) and is_primitive(g2), "split broke primitivity"
            work.extend([g1, g2])
    return tuple(sorted(irreducible))


K5 = canon(nx.complete_graph(5))


def ancestor_type(anc):
    if len(anc) >= 2:
        return "product"
    return "K5-family" if anc[0] == K5 else "prime"
