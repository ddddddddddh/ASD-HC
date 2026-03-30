import math

import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
import dgl
import json
import pdb

from sklearn.metrics import roc_auc_score


def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense(), sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    d_inv_sqrt = np.power(adj.sum(1), -0.5)
    d_mat_inv_sqrt = sp.diags(np.array(d_inv_sqrt).flatten())
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot

def load_mat(dataset, train_rate=0.8, val_rate=0):
    """Load .mat dataset."""
    # data = sio.loadmat("../data/{}.mat".format(dataset+'/'+dataset))
    data = sio.loadmat("../data/{}.mat".format(dataset))

    label = data['Label'] if ('Label' in data) else data['Lable']
    attr = data['Attributes'] if ('Attributes' in data) else data['X']
    network = data['Network'] if ('Network' in data) else data['A']

    adj = sp.csr_matrix(network)
    feat = sp.lil_matrix(attr)

    labels = np.squeeze(np.array(data['Class'], dtype=np.int64) - 1)
    num_classes = np.max(labels) + 1
    labels = dense_to_one_hot(labels, num_classes)

    ano_labels = np.squeeze(np.array(label))
    if 'str_anomaly_label' in data:
        str_ano_labels = np.squeeze(np.array(data['str_anomaly_label']))
        attr_ano_labels = np.squeeze(np.array(data['attr_anomaly_label']))
    else:
        str_ano_labels = None
        attr_ano_labels = None

    num_node = adj.shape[0]
    num_train = int(num_node * train_rate)
    num_val = int(num_node * val_rate)
    all_idx = list(range(num_node))
    random.shuffle(all_idx)
    idx_train = all_idx[ : num_train]
    idx_val = all_idx[num_train : num_train + num_val]
    idx_test = all_idx[num_train + num_val : ]

    return adj, feat, labels, idx_train, idx_val, idx_test, ano_labels, str_ano_labels, attr_ano_labels

def adj_to_dgl_graph(adj):
    """Convert adjacency matrix to dgl format."""
    # nx_graph = nx.from_scipy_sparse_matrix(adj)
    nx_graph = nx.from_numpy_array(adj)
    dgl_graph = dgl.DGLGraph(nx_graph)
    return dgl_graph

def generate_rwr_subgraph(dgl_graph, subgraph_size):
    """Generate subgraph with RWR algorithm."""
    all_idx = list(range(dgl_graph.number_of_nodes()))
    reduced_size = subgraph_size - 1
    traces = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, all_idx, restart_prob=1, max_nodes_per_seed=subgraph_size*3)

    subv = []

    for i,trace in enumerate(traces):
        subv.append(torch.unique(torch.cat(trace),sorted=False).tolist())
        retry_time = 0
        while len(subv[i]) < reduced_size:
            cur_trace = dgl.contrib.sampling.random_walk_with_restart(dgl_graph, [i], restart_prob=0.9, max_nodes_per_seed=subgraph_size*5)
            subv[i] = torch.unique(torch.cat(cur_trace[0]),sorted=False).tolist()
            retry_time += 1
            if (len(subv[i]) <= 2) and (retry_time >10):
                subv[i] = (subv[i] * reduced_size)
        subv[i] = subv[i][:reduced_size]
        subv[i].append(i)

    return subv

def k_order_neighbors(adj, k):
    '''Calculate the k-order neighbors'''
    hop1 = adj.sum(axis=1)
    ss = hop1
    while k>1:
        ss = adj.dot(ss)
        k -= 1
    return ss

def search_path(adj, len_path, ss):
    path_list = []
    for node in range(adj.shape[0]):
        path = []
        neighbors = adj[node].nonzero()[1]  # 当前节点的所有邻居节点
        len_adj = len(neighbors)
        for i in range(len_path):
            next_idx = ss[neighbors].argmax()
            next_node = neighbors[next_idx]
            while next_node in path and len_adj >= len_path:
                neighbors = np.delete(neighbors, next_idx)
                if len(neighbors) == 0:
                    print(node, i, "empty2")
                    break
                    # neighbors = adj[node].nonzero()[1]
                next_idx = ss[neighbors].argmax()
                next_node = neighbors[next_idx]
            neighbors = adj[next_node].nonzero()[1]
            path.append(next_node)
        path.append(node)  # 最终将当前节点置于最后
        path_list.append(path)

    return path_list

def DP(dgl_graph,nb_nodes,alg,alpha, dataset):
    ano_score_final = np.load("../data/" + dataset + "/ano_score_final.npy")
    #ano_label = np.load("../data/" + dataset + "/ano_label.npy")
    #pdb.set_trace()
    #auc = roc_auc_score(ano_label, ano_score_final)

    mean = np.mean(ano_score_final)
    std = np.std(ano_score_final)
    z = mean + std
    #z=0.15
    ano_nodes_dict = {i: k for i, k in enumerate(ano_score_final) if k > z}
    if len(ano_nodes_dict) == 0:
        ano_nodes_dict = {i: k for i,k in enumerate(ano_score_final) if k >= mean}
    ano_nodes = torch.tensor(list(ano_nodes_dict.keys()))
    print('number of ano_nodes:', len(ano_nodes))
    #ano_subgraphs = dgl.contrib.sampling.random_walk(dgl_graph, ano_nodes, num_traces=1, num_hops=nb_nodes)
    ano_subgraphs = dgl.contrib.sampling.random_walk(dgl_graph, ano_nodes, num_traces=1, num_hops=len(ano_nodes))
    max_tf = 0
    max_na =0
    max_subg = []
    max_na_list = []
    max_i= 0
    result = {}
    for i, nodes in enumerate(ano_subgraphs):
        subg = torch.unique(nodes, sorted=True).tolist()
        na_list = [n for n in subg if n in ano_nodes_dict]
        na = len(na_list)
        n = len(subg)
        if na ==0: continue
        if alg == 'BJ':
            tf = fai(alpha,na,n)
        elif alg == 'HC':
            tf = fai_HC(alpha, na, n)
        #'''
        #if tf > max_tf:
        if na > max_na:
            max_i = i
            max_tf = tf
            max_na = na
            max_na_list = na_list
            max_subg = subg
        #elif max_tf==tf==0 and len(na_list)>len(max_na_list):
        elif na == max_na and len(list(subg)) < len(max_subg):
            max_i = i
            max_tf = tf
            max_na_list = na_list
            max_subg = subg
        #'''
        result[i] = (tf, na, n)
        #print(alpha, tf, na, n)
    print('Idx:', max_i,'alpha:', z, 'Max fai:', max_tf, 'Na:', len(max_na_list), 'N:', len(max_subg))
    with open('../data/'+dataset+'/result_random_walk.json', 'w')as f: json.dump(result,f)

def fai_HC(alpha, N_alpha, N): # HC
    if N * alpha * (1 - alpha) != 0:
        f = (N_alpha - N * alpha) / math.sqrt(N * alpha * (1 - alpha))
    else:
        f = 0
    return f

def fai(alpha, N_alpha, N):  # BJ
    f = N * KL(N_alpha / N, alpha)
    return f

def KL(a, b):
    if a >= b and b != 0:
        if a == 1:
            return a * math.log(a / b)
        else:
            return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))
    else:
        return 0

def bfs(visited, queue, adj, nb_nodes,alg,alpha, dataset):  # function for BFS
    ano_score_final = np.load("../data/" + dataset + "/ano_score_final.npy")
    ano_label = np.load("../data/" + dataset + "/ano_label.npy")
    pvalues = Pvalue(ano_score_final, dataset)

    ano_nodes_dict = {i: k for i, k in pvalues.items() if k <= alpha}
    # g = nx.from_scipy_sparse_matrix(adj)
    g = nx.from_numpy_array(adj)

    for node in ano_nodes_dict.keys():
        visited.append(node)
        queue.append(node)

        while queue:  # Creating loop to visit each node
            m = queue.pop(0)
            #print(m, end=" ")
            for neighbour in g.neighbors(m):
                if neighbour not in visited:
                    visited.append(neighbour)
                    queue.append(neighbour)
    visited = list(set(visited))

    sb = nx.induced_subgraph(g, visited)
    c = max(nx.connected_components(sb))
    na_list = [n for n in c if n in ano_nodes_dict]
    na = len(na_list)
    n = len(c)

    max_score = [ano_score_final[x] for x in c]
    max_label = [ano_label[x] for x in c]
    max_ano_score = [ano_score_final[x] for x in na_list]
    max_ano_label = [ano_label[x] for x in na_list]
    auc = roc_auc_score(max_label, max_score)
    ano_auc = roc_auc_score(max_ano_label, max_ano_score)
    print("max_sub: auc", auc,"max_ano_sub: auc", ano_auc)

    if alg == 'BJ':
        tf = fai(alpha, na, n)
    elif alg == 'HC':
        tf = fai_HC(alpha, na, n)
    print("tf:", tf, "na:", na, "n:", n)
    return

def Pvalue(ano_score_final, dataset):
    _dict = {}
    v = ano_score_final
    Total = len(v)
    for i in range(Total):
        dist = [t for t in v if t > v[i]]
        c = len(dist)
        p = c/Total
        _dict[i] = round(p, 5)
    with open('../data/{}/ASD-HC Pvalue-a.json'.format(dataset), 'w')as f: json.dump(_dict, f)
    print('ASD-HC Pvalue-a.json'," is :", len(_dict))
    return _dict


