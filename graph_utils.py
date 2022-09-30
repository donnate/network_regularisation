import numpy as np
import numpy.linalg as la
import scipy


def incidence_matrix(G, nodelist=None, edgelist=None, oriented=False, weight=None):
    """Returns incidence matrix of G.

    The incidence matrix assigns each row to a node and each column to an edge.
    For a standard incidence matrix a 1 appears wherever a row's node is
    incident on the column's edge.  For an oriented incidence matrix each
    edge is assigned an orientation (arbitrarily for undirected and aligning to
    direction for directed).  A -1 appears for the source (tail) of an edge and
    1 for the destination (head) of the edge.  The elements are zero otherwise.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    nodelist : list, optional   (default= all nodes in G)
       The rows are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    edgelist : list, optional (default= all edges in G)
       The columns are ordered according to the edges in edgelist.
       If edgelist is None, then the ordering is produced by G.edges().

    oriented: bool, optional (default=False)
       If True, matrix elements are +1 or -1 for the head or tail node
       respectively of each edge.  If False, +1 occurs at both nodes.

    weight : string or None, optional (default=None)
       The edge data key used to provide each value in the matrix.
       If None, then each edge has weight 1.  Edge weights, if used,
       should be positive so that the orientation can provide the sign.

    Returns
    -------
    A : SciPy sparse matrix
      The incidence matrix of G.

    Notes
    -----
    For MultiGraph/MultiDiGraph, the edges in edgelist should be
    (u,v,key) 3-tuples.

    "Networks are the best discrete model for so many problems in
    applied mathematics" [1]_.

    References
    ----------
    .. [1] Gil Strang, Network applications: A = incidence matrix,
       http://videolectures.net/mit18085f07_strang_lec03/
    """
    import scipy as sp
    import scipy.sparse  # call as sp.sparse

    if nodelist is None:
        nodelist = list(G)
    if edgelist is None:
        if G.is_multigraph():
            edgelist = list(G.edges(keys=True))
        else:
            edgelist = list(G.edges())
    A = scipy.sparse.lil_matrix((len(nodelist), len(edgelist)))
    node_index = {node: i for i, node in enumerate(nodelist)}
    for ei, e in enumerate(edgelist):
        (u, v) = e[:2]
        if u == v:
            continue  # self loops give zero column
        try:
            ui = node_index[u]
            vi = node_index[v]
        except KeyError as err:
            raise nx.NetworkXError(
                f"node {u} or {v} in edgelist but not in nodelist"
            ) from err
        if weight is None:
            wt = 1
        else:
            if G.is_multigraph():
                ekey = e[2]
                wt = G[u][v][ekey].get(weight, 1)
            else:
                wt = G[u][v].get(weight, 1)
        if oriented:
            A[ui, ei] = -wt
            A[vi, ei] = wt
        else:
            A[ui, ei] = wt
            A[vi, ei] = wt
    import warnings

    warnings.warn(
        "incidence_matrix will return a scipy.sparse array instead of a matrix in Networkx 3.0.",
        FutureWarning,
        stacklevel=2,
    )
    # TODO: Rm sp.sparse.csc_matrix in Networkx 3.0
    return A.asformat("csc")

def return_incidence(p):
    return np.asarray((incidence_matrix(nx.complete_graph(p), oriented = True)).T.todense())


def cor_from_G(G, a): ### Covariance is inverse of L + a*I
    L = nx.laplacian_matrix(G)
    p = L.shape[0]
    C = la.inv(L.todense() + a*np.identity(p))
    W = np.sqrt(np.diag(np.diagonal(C)))
    C_1 = la.inv(W)@C@la.inv(W)
    return C_1

def cov_from_G(G, a): ### Covariance is inverse of L + a*I
    L = nx.laplacian_matrix(G)
    p = L.shape[0]
    C = la.inv(L.todense() + a*np.identity(p))
    return C

def D_from_G(G):
    return np.asarray(incidence_matrix(G, oriented = True).T.todense())
