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

if torch.cuda.is_available():
    print('Using CUDA')
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

if torch.cuda.is_available():
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]).cuda())
else:
    b_xent = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.negsamp_ratio]))

cnt_wait = 0
best = 1e9
best_t = 0
batch_num = nb_nodes // batch_size +1

#
# Train model
with tqdm(total=args.num_epoch) as pbar:
    pbar.set_description('Training')
    for epoch in range(args.num_epoch):

        loss_full_batch = torch.zeros((nb_nodes,1))
        if torch.cuda.is_available():
            loss_full_batch = loss_full_batch.cuda()

        model.train()

        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)
        total_loss = 0.

        for batch_idx in range(batch_num):

            optimiser.zero_grad()

            is_final_batch = (batch_idx == (batch_num - 1))

            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                idx = all_idx[batch_idx * batch_size:]

            cur_batch_size = len(idx)

            lbl = torch.unsqueeze(torch.cat((torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio))), 1)

            if torch.cuda.is_available():
                lbl = lbl.cuda()

            ba = []
            bf = []
            added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
            added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
            added_adj_zero_col[:, -1, :] = 1.
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

            if torch.cuda.is_available():
                lbl = lbl.cuda()
                added_adj_zero_row = added_adj_zero_row.cuda()
                added_adj_zero_col = added_adj_zero_col.cuda()
                added_feat_zero_row = added_feat_zero_row.cuda()

            for i in idx:
                cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                cur_feat = features[:, subgraphs[i], :]
                ba.append(cur_adj)
                bf.append(cur_feat)

            ba = torch.cat(ba)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)
            bf = torch.cat(bf)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]),dim=1)

            logits = model(bf, ba)
            loss_all = b_xent(logits, lbl)

            loss = torch.nanmean(loss_all)

            loss.backward()
            optimiser.step()

            loss = loss.detach().cpu().numpy()

            loss_full_batch[idx] = loss_all[: cur_batch_size].detach()

            if not is_final_batch:
                total_loss += loss

            mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes

            if mean_loss < best:
                best = mean_loss
                best_t = epoch
                cnt_wait = 0
                torch.save(model.state_dict(), '../data/'+args.dataset+'/best_model.pkl')
            else:
                cnt_wait += 1

        pbar.set_postfix(loss=mean_loss)
        pbar.update(1)

# Test model
print('Loading {}th epoch'.format(best_t))
model.load_state_dict(torch.load('../data/'+args.dataset+'/best_model.pkl'))

multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))
multi_round_ano_score_p = np.zeros((args.auc_test_rounds, nb_nodes))
multi_round_ano_score_n = np.zeros((args.auc_test_rounds, nb_nodes))

with tqdm(total=args.auc_test_rounds) as pbar_test:
    pbar_test.set_description('Testing')
    for round in range(args.auc_test_rounds):

        all_idx = list(range(nb_nodes))
        random.shuffle(all_idx)

        for batch_idx in range(batch_num):

            optimiser.zero_grad()

            is_final_batch = (batch_idx == (batch_num - 1))

            if not is_final_batch:
                idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            else:
                idx = all_idx[batch_idx * batch_size:]

            cur_batch_size = len(idx)

            ba = []
            bf = []
            added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size))
            added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1))
            added_adj_zero_col[:, -1, :] = 1.
            added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size))

            if torch.cuda.is_available():
                lbl = lbl.cuda()
                added_adj_zero_row = added_adj_zero_row.cuda()
                added_adj_zero_col = added_adj_zero_col.cuda()
                added_feat_zero_row = added_feat_zero_row.cuda()

            for i in idx:
                cur_adj = adj[:, subgraphs[i], :][:, :, subgraphs[i]]
                cur_feat = features[:, subgraphs[i], :]
                ba.append(cur_adj)
                bf.append(cur_feat)

            ba = torch.cat(ba)
            ba = torch.cat((ba, added_adj_zero_row), dim=1)
            ba = torch.cat((ba, added_adj_zero_col), dim=2)
            bf = torch.cat(bf)
            bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

            with torch.no_grad():
                logits = torch.squeeze(model(bf, ba))
                logits = torch.nan_to_num(logits)
                logits = torch.sigmoid(logits)

            ano_score = - (logits[:cur_batch_size] - logits[cur_batch_size:])
            multi_round_ano_score[round, idx] = ano_score.cpu().numpy()
        pbar_test.update(1)

#pdb.set_trace()
ano_score_final = np.nanmean(multi_round_ano_score, axis=0)
# np.save("../data/"+args.dataset+"/ano_score_final.npy", ano_score_final)
# np.save("../data/"+args.dataset+"/ano_label.npy", ano_label)
#
auc = roc_auc_score(ano_label, ano_score_final)
print('AUC:{:.4f}'.format(auc))
#
# DP(dgl_graph,nb_nodes,'HC',0.15, args.dataset)

# """
# ano_score_final = np.load("../data/" + args.dataset + "/ano_score_final.npy")
# ano_label = np.load("../data/" + args.dataset + "/ano_label.npy")
# auc = roc_auc_score(ano_label, ano_score_final)
# print('AUC:{:.4f}'.format(auc))
# DP(dgl_graph,nb_nodes,'HC',0.15, args.dataset)
#
# print("Following is the Breadth-First Search")
# visited = []  # List for visited nodes.
# queue = []  # Initialize a queue
# bfs(visited, queue, adj_o, nb_nodes, 'HC', 0.15, args.dataset)

