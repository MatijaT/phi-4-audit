"""Independent check of the labels: the c2 invariant by brute-force point counting.

Psi_G(x) = sum over spanning trees T of prod_{e not in T} x_e (Kirchhoff polynomial of the
decompleted graph G) equals det(C diag(x) C^T) for a cycle-space basis C. With
[Psi_G]_p = #{x in F_p^N : Psi_G(x) = 0}, the c2 invariant is c2(G)_p = [Psi_G]_p / p^2 mod p.
Exponential in the number of edges, so only usable for small graphs and small primes; the
labels in this repository come from the Periods file, and this module spot-checks them.
"""
import numpy as np

def cycle_matrix(edges, V):
    """edges: list of (u,v), vertices 0..V-1. Returns C (L x N int) of fundamental cycles."""
    N=len(edges)
    par=list(range(V))
    def find(a):
        while par[a]!=a: par[a]=par[par[a]]; a=par[a]
        return a
    treeedges=[]; nontree=[]; adj=[[] for _ in range(V)]
    for ei,(u,v) in enumerate(edges):
        ru,rv=find(u),find(v)
        if ru!=rv:
            par[ru]=rv; treeedges.append(ei); adj[u].append((v,ei)); adj[v].append((u,ei))
        else:
            nontree.append(ei)
    L=len(nontree)
    C=np.zeros((L,N),dtype=np.int64)
    # for each non-tree edge, find tree path between its endpoints (BFS), set cycle row
    import collections
    for ci,ei in enumerate(nontree):
        u,v=edges[ei]
        C[ci,ei]=1
        # BFS from v to u in tree, record edges with orientation
        prev={v:(None,None,None)}
        dq=collections.deque([v]); found=False
        while dq and not found:
            x=dq.popleft()
            for (y,te) in adj[x]:
                if y not in prev:
                    prev[y]=(x,te,edges[te]); dq.append(y)
                    if y==u: found=True; break
        # walk back u->v
        x=u
        while prev[x][0] is not None:
            px,te,(a,b)=prev[x]
            # edge te is (a,b); in the cycle we traverse from px to x. orientation +1 if a==px (a->b == px->x)
            C[ci,te]= 1 if (a==px and b==x) else -1
            x=px
    return C

_INV={}
def inv_table(p):
    if p not in _INV:
        t=np.zeros(p,dtype=np.int64)
        for a in range(1,p): t[a]=pow(a,p-2,p)
        _INV[p]=t
    return _INV[p]

def _count_singular(M,p,inv):
    """M: (B,L,L) int mod p. Return count of singular (det==0 mod p) via batched elimination."""
    B,L,_=M.shape
    M=(M%p).astype(np.int64)
    idx=np.arange(B)
    singular=np.zeros(B,dtype=bool)
    for c in range(L):
        sub=M[:,c:,c]                      # (B, L-c)
        nz=sub!=0
        has=nz.any(axis=1)
        singular |= ~has                   # no pivot in this column -> singular
        active= has & ~singular            # (has==True here means not singular this step)
        if not active.any(): continue
        off=nz.argmax(axis=1)              # first nonzero offset (valid where has)
        prow=c+off
        a=idx[active]; pr=prow[active]
        # swap row c <-> prow for active
        tmp=M[a,c,:].copy()
        M[a,c,:]=M[a,pr,:]
        M[a,pr,:]=tmp
        piv=M[a,c,c]                        # nonzero
        pivinv=inv[piv]
        # eliminate rows below c
        for r in range(c+1,L):
            factor=(M[a,r,c]*pivinv)%p
            nzf=factor!=0
            if nzf.any():
                aa=a[nzf]; f=factor[nzf][:,None]
                M[aa,r,:]=(M[aa,r,:]-f*M[aa,c,:])%p
    return int(singular.sum())

def point_count(C,p,ceiling=10**8,batch=400000):
    """count x in F_p^N with Psi(x)=det(C diag(x) C^T)=0 mod p. None if p^N>ceiling."""
    L,N=C.shape
    total=p**N
    if total>ceiling: return None
    inv=inv_table(p)
    Cm=(C%p).astype(np.int64)
    R=np.zeros((N,L,L),dtype=np.int64)     # R[e]=col_e outer col_e
    for e in range(N):
        col=Cm[:,e]; R[e]=np.outer(col,col)%p
    Rf=R.reshape(N,L*L)
    zeros=0
    pw=p**np.arange(N)                     # base-p decode
    start=0
    while start<total:
        b=min(batch,total-start)
        ids=np.arange(start,start+b)
        x=((ids[:,None]//pw)%p).astype(np.int64)   # (b,N)
        Mf=(x@Rf)%p                                 # (b,L*L)
        M=Mf.reshape(b,L,L)
        zeros+=_count_singular(M,p,inv)
        start+=b
    return zeros

def c2_at_prime(edges,V,p,ceiling=10**8):
    C=cycle_matrix(edges,V); N=C.shape[1]
    cnt=point_count(C,p,ceiling)
    if cnt is None: return ('TOO_LARGE',None)
    if cnt % (p*p)!=0: return ('NOT_DIV_P2',cnt)   # should not happen for valid graphs
    return (cnt,(cnt//(p*p))%p)

# ---- spanning tree count (Kirchhoff) for sanity / a feature ----
def n_spanning_trees(edges,V):
    Lap=np.zeros((V,V),dtype=np.int64)
    for (u,v) in edges:
        Lap[u,u]+=1; Lap[v,v]+=1; Lap[u,v]-=1; Lap[v,u]-=1
    M=Lap[1:,1:].astype(float)
    return int(round(abs(np.linalg.det(M))))


def c2(G_completed, p, ceiling=10**8):
    """c2 at p (as a residue 0..p-1) of a completed graph, via the decompletion at vertex 0."""
    import networkx as nx
    G = nx.convert_node_labels_to_integers(G_completed)
    H = G.copy()
    H.remove_node(0)
    H = nx.convert_node_labels_to_integers(H)
    cnt, val = c2_at_prime(list(H.edges()), H.number_of_nodes(), p, ceiling)
    if not isinstance(cnt, int):
        raise ValueError(f"point count {cnt} at p={p}")
    return val
