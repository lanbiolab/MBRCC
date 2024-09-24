import random
import torch
import math
import numpy as np
import multiprocessing
import heapq
from time import time
from utils.metrics import *
from utils.parser import parse_args


args = parse_args()
Ks = eval(args.Ks)
device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
BATCH_SIZE = args.test_batch_size
batch_test_flag = args.batch_test_flag


def test(model, user_dict, n_params):

    global n_users, n_items
    n_items = n_params['n_items']
    n_users = n_params['n_users']

    global test_user_set
    test_user_set = user_dict['test_user_set']

    u_res, i_res = model.generate()

    scores = torch.mm(u_res, i_res.t())
    K_list = args.TopK
    racall_list = []
    ndcg_list = []

    for k in K_list:
        _, rating_K = torch.topk(scores, k=k)
        c = 0
        ndcg_tem = 0
        for i in test_user_set.keys():
            r = [0 for j in range(k)]
            if int(test_user_set[i][0]) in rating_K[i]:
                rank = rating_K[i].tolist()
                index = rank.index(int(test_user_set[i][0]))
                r[index] = 1
                c = c + 1
            this_ndcg = ndcg_at_k(r, k, test_user_set[i][0], 1)
            ndcg_tem = ndcg_tem + this_ndcg
        recall = c / len(test_user_set)
        ndcg = ndcg_tem / len(test_user_set)
        racall_list.append(recall)
        ndcg_list.append(ndcg)
    return racall_list, ndcg_list


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k] #取前k个
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.

def ndcg_at_k(r, k, ground_truth, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain

        Low but correct defination
    """

    GT = set([ground_truth])
    if len(GT) > k :
        sent_list = [1.0] * k
    else:
        sent_list = [1.0]*len(GT) + [0.0]*(k-len(GT))
    dcg_max = dcg_at_k(sent_list, k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max