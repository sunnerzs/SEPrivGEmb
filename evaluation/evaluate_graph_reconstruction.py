import pickle
from evaluation import metrics
from utils import evaluation_util, graph_util
import networkx as nx
import numpy as np


def evaluateStaticGraphReconstruction(digraph,
                                      X_st, node_l=None, file_suffix=None,
                                      sample_ratio_e=0.01, is_undirected=True,
                                      is_weighted=False):
    node_num = len(digraph.nodes())
    # evaluation
    if sample_ratio_e:
        eval_edge_pairs = evaluation_util.get_random_edge_pairs(
            node_num,
            sample_ratio_e,
            is_undirected
        )
    else:
        eval_edge_pairs = None
    if file_suffix is None:
        estimated_adj = get_reconstructed_adj(X_st, node_l)
    else:
        estimated_adj = get_reconstructed_adj(
            X_st,
            file_suffix,
            node_l
        )
    predicted_edge_list = evaluation_util.get_edge_list_from_adj_mtrx(
        estimated_adj,
        is_undirected=is_undirected,
        edge_pairs=eval_edge_pairs
    )
    MAP = metrics.computeMAP(predicted_edge_list, digraph, is_undirected=is_undirected)
    prec_curv, _ = metrics.computePrecisionCurve(predicted_edge_list, digraph)
    # If weighted, compute the error in reconstructed weights of observed edges
    # if is_weighted:
    #     digraph_adj = nx.to_numpy_matrix(digraph)
    #     estimated_adj[digraph_adj == 0] = 0
    #     err = np.linalg.norm(digraph_adj - estimated_adj)
    #     err_baseline = np.linalg.norm(digraph_adj)
    # else:
    #     err = None
    #     err_baseline = None
    # return MAP, prec_curv, err, err_baseline
    return MAP, prec_curv

def get_reconstructed_adj(X_st, node_l=None):
    """Compute the adjacency matrix from the learned embedding

    Returns:
        A numpy array of size #nodes * #nodes containing the reconstructed adjacency matrix.
    """
    global X_start, X_end
    # print(len(X_st))
    if len(X_st) == 1:
        X_start = X_st[0]
        X_end = X_st[0]
    if len(X_st) == 2:
        X_start = X_st[0]
        X_end = X_st[1]

    node_num = X_start.shape[0]
    # if X_st is not None:
    #     node_num = X_st[0].shape[0]
    #     # self._X = X
    # else:
    #     node_num = X_st.shape[0]
    adj_mtx_r = np.zeros((node_num, node_num))
    for v_i in range(node_num):
        for v_j in range(node_num):
            if v_i == v_j:
                continue
            # adj_mtx_r[v_i, v_j] = get_edge_weight(v_i, v_j)
            adj_mtx_r[v_i, v_j] = np.dot(X_start[v_i], X_end[v_j])
    return adj_mtx_r

