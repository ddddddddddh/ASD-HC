"""
FileName: 
Author: 
Version: 
Date: 2025/2/2216:14
Description: 
"""
import os
import dgl
import pdb
import random
import argparse
import torch
import torch.nn as nn
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, roc_curve
from model import *
from utils import *
from tqdm import tqdm

# Set argument
parser = argparse.ArgumentParser(description='Self-Supervised Contrastive Learning-based Anomaly Subgraph Detection')
parser.add_argument('--dataset', type=str, default='cora')  # 'BlogCatalog'  'Flickr'  'ACM'  'cora'  'citeseer'  'pubmed'
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--drop_prob', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--readout', type=str, default='avg')  # max min avg  weighted_sum
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio', type=int, default=1)
parser.add_argument('--t', type=int, default=15) # the optimum path length
parser.add_argument('--k', type=int, default=3) # k-order neighbor

args = parser.parse_args()

if args.lr is None:
    if args.dataset in ['cora','citeseer','pubmed','Flickr']:
        args.lr = 1e-3
    elif args.dataset in ['ACM','Weibo','BC','email']:
        args.lr = 5e-4
    elif args.dataset == 'BlogCatalog':
        args.lr = 3e-3

if args.num_epoch is None:
    if args.dataset in ['cora','citeseer','pubmed']:
        args.num_epoch = 100
    elif args.dataset in ['BlogCatalog','Flickr','ACM','Weibo','BC','email']:
        args.num_epoch = 200

batch_size = args.batch_size
subgraph_size = args.t

# Set random seed
dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
os.environ['OMP_NUM_THREADS'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Load and preprocess data
adj, features, labels, idx_train, idx_val, idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)

adj_o = adj
features, _ = preprocess_features(features)

dgl_graph = adj_to_dgl_graph(adj)

nb_nodes = features.shape[0]
ft_size = features.shape[1]
# """
adj = normalize_adj(adj)
adj = (adj + sp.eye(adj.shape[0])).todense()
hopk = k_order_neighbors(adj, args.k)
subgraphs = search_path(adj, subgraph_size-1, hopk)

features = torch.FloatTensor(features[np.newaxis])
adj = torch.FloatTensor(adj[np.newaxis])
labels = torch.FloatTensor(labels[np.newaxis])
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio, args.readout)
optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# epoch100, 0.9239
# epoch400, 0.9020

model.load_state_dict(torch.load('../data/'+args.dataset+'/best_model.pkl'))
model.eval()
with torch.no_grad():
    emd = model.analyse(features, adj)


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# (1, 2708, 64) -> (2708, 64)
emd = emd.squeeze(0)
torch.save(emd, '../data/'+args.dataset+'/ASD_HC_node_emd.pt')
# 使用TSNE将64降维到2维
tsne_emd = TSNE(n_components=2, random_state=42)
emd_2d = tsne_emd.fit_transform(emd)

# 绘图
colors = ['red' if label == 1 else 'blue' for label in ano_label]
plt.figure(figsize=(8, 6))
plt.scatter(emd_2d[:, 0], emd_2d[:, 1], c=colors, s=10)
plt.show()