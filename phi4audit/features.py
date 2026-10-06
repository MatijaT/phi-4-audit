"""24 elementary invariants of the completed graph. All are isomorphism-invariant."""
import igraph
import networkx as nx
import numpy as np

FEATURE_KEYS = ['L', 'Vcomp', 'aut', 'log_aut', 'ntrees', 'log_ntrees', 'triangles',
                'double_tri', 'C4', 'girth', 'diameter', 'planar', 'radius', 'edge_conn',
                'vertex_conn', 'ev2', 'ev_min', 'spec_gap', 'lap2', 'lap_max',
                'n_distinct_ev', 'bipartite', 'trace_A3', 'trace_A4']


def aut_order(G):
    nodes = sorted(G.nodes())
    idx = {u: i for i, u in enumerate(nodes)}
    g = igraph.Graph(n=len(nodes), edges=[(idx[u], idx[v]) for u, v in G.edges()])
    return int(g.count_automorphisms())


def spanning_trees_decompleted(G):
    """Spanning trees of the decompleted graph G - v, i.e. the number of monomials of its
    Kirchhoff polynomial. Depends on v when G is not vertex-transitive; we take the
    minimum over v so that the feature is an isomorphism invariant."""
    best = None
    for v in G.nodes():
        H = G.copy()
        H.remove_node(v)
        Lap = nx.laplacian_matrix(H, nodelist=sorted(H.nodes())).toarray().astype(float)
        t = int(round(abs(np.linalg.det(Lap[1:, 1:]))))
        best = t if best is None else min(best, t)
    return best


def double_triangles(A):
    """# pairs of triangles sharing an edge = sum over edges of C(common neighbours, 2)."""
    tot = 0
    n = A.shape[0]
    for u in range(n):
        for v in range(u + 1, n):
            if A[u, v]:
                t = int(np.dot(A[u], A[v]))
                tot += t * (t - 1) // 2
    return tot


def features(G):
    G = nx.convert_node_labels_to_integers(G)
    V = G.number_of_nodes()
    A = nx.to_numpy_array(G, nodelist=range(V))
    ev = np.sort(np.linalg.eigvalsh(A))
    lev = np.sort(np.linalg.eigvalsh(np.diag(A.sum(1)) - A))
    A2 = A @ A
    aut = aut_order(G)
    nt = spanning_trees_decompleted(G)
    return {
        'L': V - 2, 'Vcomp': V, 'aut': aut, 'log_aut': float(np.log(aut)),
        'ntrees': nt, 'log_ntrees': float(np.log(max(nt, 1))),
        'triangles': sum(nx.triangles(G).values()) // 3,
        'double_tri': double_triangles(A.astype(np.int64)),
        'C4': int(round((np.trace(A2 @ A2) - 2 * np.sum(A2) + np.sum(A)) / 8)),
        'girth': nx.girth(G),
        'diameter': nx.diameter(G), 'planar': int(nx.check_planarity(G)[0]),
        'radius': nx.radius(G),
        'edge_conn': nx.edge_connectivity(G), 'vertex_conn': nx.node_connectivity(G),
        'ev2': float(ev[-2]), 'ev_min': float(ev[0]), 'spec_gap': float(4 - ev[-2]),
        'lap2': float(lev[1]), 'lap_max': float(lev[-1]),
        'n_distinct_ev': len(np.unique(np.round(ev, 6))),
        'bipartite': int(nx.is_bipartite(G)),
        'trace_A3': float(np.trace(A2 @ A)), 'trace_A4': float(np.trace(A2 @ A2)),
    }
