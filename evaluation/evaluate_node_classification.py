try: import cPickle as pickle
except: import pickle
from sklearn import model_selection as sk_ms
from sklearn.multiclass import OneVsRestClassifier as oneVr
from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
import operator
import numpy as np

# class TopKRanker(oneVr):
#     def predict(self, X, top_k_list):
#         assert X.shape[0] == len(top_k_list)
#         probs = np.asarray(super(TopKRanker, self).predict_proba(X))
#         prediction = np.zeros((X.shape[0], self.classes_.shape[0]))
#         for i, k in enumerate(top_k_list):
#             probs_ = probs[i, :]
#             labels = self.classes_[probs_.argsort()[-int(k):]].tolist()
#             for label in labels:
#                 prediction[i, label] = 1
#         return prediction
#
#
# def evaluateNodeClassification(X, Y, test_ratio):
#     X_train, X_test, Y_train, Y_test = sk_ms.train_test_split(
#         X,
#         Y,
#         test_size=test_ratio
#     )
#     top_k_list = Y_test
#     # try:
#     #     top_k_list = list(Y_test.toarray().sum(axis=1))
#     # except:
#     #     top_k_list = list(Y_test.sum(axis=1))
#     classif2 = TopKRanker(lr())
#     classif2.fit(X_train, Y_train)
#     prediction = classif2.predict(X_test, top_k_list)
#     micro = f1_score(Y_test, prediction, average='micro')
#     macro = f1_score(Y_test, prediction, average='macro')
#     return (micro, macro)

# class TopKRanker(oneVr):
#     def predict(self, X, top_k_list):
#         assert X.shape[0] == len(top_k_list)
#         probs = np.asarray(super(TopKRanker, self).predict_proba(X))
#         prediction = np.zeros((X.shape[0], self.classes_.shape[0]))
#         for i, k in enumerate(top_k_list):
#             probs_ = probs[i, :]
#             labels = self.classes_[probs_.argsort()[-int(k):]].tolist()
#             for label in labels:
#                 prediction[i, label] = 1
#         return prediction
# class NodeClassification():
#     def __init__(self):
#         self.node_label = {}
#         with open(config.test_filename) as infile:
#             for line in infile.readlines():
#                 line = line.strip()
#                 line = line.split(',')
#                 s = int(line[0])
#                 label = int(line[1])
#                 self.node_label[s] = label

def evaluateNodeClassification(embedding_matrix, node_label=None, test_ratio=None):
    # node_label = {}
    # with open(test_filename) as infile:
    #     for line in infile.readlines():
    #         if operator.contains(line, ','):
    #             line = line.strip().split(',')
    #         else:
    #             line = line.strip().split()
    #         s = int(line[0])
    #         label = int(line[1])
    #         node_label[s] = label

    embedding_list = embedding_matrix.tolist()
    X = []
    Y = []
    for t in node_label:
        X.append(embedding_list[t])
        # X.append(embedding_list[t] + embedding_list[t])
        Y.append(node_label[t])

    X_train, X_test, Y_train, Y_test = sk_ms.train_test_split(
        X,
        Y,
        test_size=test_ratio
    )

    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    # print('执行了 fit')
    Y_pred = lr.predict(X_test)
    # print('没执行 predict')
    micro_f1 = f1_score(Y_test, Y_pred, average='micro')
    macro_f1 = f1_score(Y_test, Y_pred, average='macro')

    return micro_f1, macro_f1
