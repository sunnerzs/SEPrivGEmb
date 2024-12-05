try: import cPickle as pickle
except: import pickle
from evaluation import metrics
from utils import evaluation_util, graph_util
import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score
from networkx import weakly_connected_component_subgraphs

def evaluateStaticLinkPrediction(digraph, graph_embedding,
                                 train_ratio=0.8,
                                 n_sample_nodes=None,
                                 sample_ratio_e=None,
                                 no_python=False,
                                 is_undirected=True):
    node_num = digraph.number_of_nodes()
    # seperate train and test graph
    train_digraph, test_digraph = evaluation_util.split_di_graph_to_train_test(
        digraph,
        train_ratio=train_ratio,
        is_undirected=is_undirected
    )
    if not nx.is_connected(train_digraph.to_undirected()):
        # Here Networkx is version 1.11, others may yield error
        train_digraph = max(
            nx.weakly_connected_component_subgraphs(train_digraph),
            key=len
        )
        # largest = max(
        #     nx.connected_components(train_digraph),
        #     key=len
        # )
        # train_digraph = train_digraph.subgraph(largest)

        tdl_nodes = train_digraph.nodes()
        nodeListMap = dict(zip(tdl_nodes, range(len(tdl_nodes))))
        nx.relabel_nodes(train_digraph, nodeListMap, copy=False)
        test_digraph = test_digraph.subgraph(tdl_nodes)
        nx.relabel_nodes(test_digraph, nodeListMap, copy=False)

    # learning graph embedding
    X, Y = graph_embedding.learn_embedding(train_digraph, test_digraph)
    # ----- test --------
    # print(X[0])
    # print(X[1])
    # -------------------
    # test_y = []
    # pred_y = []
    # node_num = train_digraph.number_of_nodes()
    # for v_i in range(node_num):
    #     for v_j in range(node_num):
    #         if v_i == v_j:
    #             continue
    #         try:
    #             # print(test_digraph[v_i][v_j]['weight'])
    #             if test_digraph[v_i][v_j]['weight'] == 1:
    #                 test_y.append(1)
    #         except:
    #             test_y.append(0)
    #         pred_y.append(X[v_i].dot(X[v_j]))
    #
    # auc = roc_auc_score(test_y, pred_y)
    # return auc

# def evaluate(train_digraph, test_digraph, X, Y):
#     test_y = []
#     pred_y = []
#     node_num = train_digraph.number_of_nodes()
#     for v_i in range(node_num):
#         for v_j in range(node_num):
#             if v_i == v_j:
#                 continue
#             try:
#                 # print(test_digraph[v_i][v_j]['weight'])
#                 if test_digraph[v_i][v_j]['weight'] == 1:
#                     test_y.append(1)
#             except:
#                 test_y.append(0)
#             pred_y.append(X[v_i].dot(Y[v_j]))
#
#     auc = roc_auc_score(test_y, pred_y)
#     return auc