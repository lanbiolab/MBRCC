import numpy as np
import scipy.sparse as sp
import pickle
from collections import defaultdict
import warnings
import torch
warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
dataset = ''
train_user_set_p = defaultdict(list)
train_user_set_c = defaultdict(list)
train_user_set_v = defaultdict(list)
test_user_set = defaultdict(list)
valid_user_set = defaultdict(list)

def read_cf(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        if 'None' in tmps:
            continue
        else:
            inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
    return np.array(inter_mat)

def statistics(train_data_p, train_data_c, train_data_v, valid_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data_p[:, 0]), max(valid_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data_p[:, 1]), max(valid_data[:, 1]), max(test_data[:, 1])) + 1

    for u_id, i_id in train_data_p:
        train_user_set_p[int(u_id)].append(int(i_id))
    for u_id, i_id in train_data_c:
        train_user_set_c[int(u_id)].append(int(i_id))
    for u_id, i_id in train_data_v:
        train_user_set_v[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in valid_data:
        valid_user_set[int(u_id)].append(int(i_id))


def build_sparse_graph(data_cf):
    def _bi_norm_lap(adj):
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    cf = data_cf.copy()
    cf[:, 1] = cf[:, 1] + n_users
    cf_ = cf.copy()
    cf_[:, 0], cf_[:, 1] = cf[:, 1], cf[:, 0]
    cf_ = np.concatenate([cf, cf_], axis=0)

    vals = [1.] * len(cf_)
    mat = sp.coo_matrix((vals, (cf_[:, 0], cf_[:, 1])), shape=(n_users+n_items, n_users+n_items))
    return _bi_norm_lap(mat)

def load_data(model_args):
    global args, dataset
    args = model_args
    dataset = args.dataset

    print('reading train and test user-item set ...')
    directory = args.data_path + dataset + '/'
    if args.dataset == "Yelp":
        train_p = read_cf(directory + 'trn_pos.txt')
        train_c = read_cf(directory + 'trn_neutral.txt')
        train_v = read_cf(directory + 'trn_tip.txt')
    else:
        train_p = read_cf(directory + 'train.txt')
        train_c = read_cf(directory + 'cart.txt')
        train_v = read_cf(directory + 'pv.txt')
    test_cf = read_cf(directory + 'test.txt')
    valid_cf = test_cf
    statistics(train_p, train_c, train_v, valid_cf, test_cf)

    print('building the adj mat ...')
    norm_mat_p = build_sparse_graph(train_p)
    norm_mat_c = build_sparse_graph(train_c)
    norm_mat_v = build_sparse_graph(train_v)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
    }
    print(f"n_users:{n_users}, n_items:{n_items}")

    pkfile = open(directory + "type_num.txt", 'rb+')
    type_num = pickle.load(pkfile)
    user_dict = {
        'train_user_set_p': train_user_set_p,
        'train_user_set_c': train_user_set_c,
        'train_user_set_v': train_user_set_v,
        'valid_user_set': None,
        'test_user_set': test_user_set,
    }
    print('loading over ...')
    return train_p, user_dict, n_params, norm_mat_p, norm_mat_c, norm_mat_v, torch.tensor(type_num)
