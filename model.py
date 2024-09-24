import math

import torch
import torch.nn as nn
from utils.contrast import Contrast
from utils.cluster import Cluster
from utils.cluster_1 import ClusterLoss1
import numpy as np
from utils.cluster_contrast import ClusterLoss, InstanceLoss


class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, n_hops, n_users, interact_mat,
                 edge_dropout_rate=0.2, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.interact_mat = interact_mat
        self.n_users = n_users
        self.n_hops = n_hops
        self.edge_dropout_rate = edge_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate

        self.dropout = nn.Dropout(p=mess_dropout_rate)

    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = x._nnz()  #
        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))

    def forward(self, user_embed, item_embed,
                mess_dropout=True, edge_dropout=True):

        all_embed = torch.cat([user_embed, item_embed], dim=0)
        agg_embed = all_embed
        embs = [all_embed]

        for hop in range(self.n_hops):
            interact_mat = self._sparse_dropout(self.interact_mat,
                                                self.edge_dropout_rate) if edge_dropout \
                                                                        else self.interact_mat

            agg_embed = torch.sparse.mm(interact_mat, agg_embed)
            if mess_dropout:
                agg_embed = self.dropout(agg_embed)
            embs.append(agg_embed)
        embs = torch.stack(embs, dim=1)
        return embs[:self.n_users, :], embs[self.n_users:, :]


class LightGCN(nn.Module):
    def __init__(self, data_config, args_config, adj_mat_p, adj_mat_c, adj_mat_v):
        super(LightGCN, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat_p = adj_mat_p
        self.adj_mat_c = adj_mat_c
        self.adj_mat_v = adj_mat_v

        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.edge_dropout = args_config.edge_dropout
        self.edge_dropout_rate = args_config.edge_dropout_rate
        self.pool = args_config.pool
        self.n_negs = args_config.n_negs
        self.ns = args_config.ns
        self.K = args_config.K
        self.tua0 = args_config.tua0
        self.tua1 = args_config.tua1
        self.tua2 = args_config.tua2
        self.tua3 = args_config.tua3
        self.lamda = args_config.lamda
        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")

        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)

        self.gcn_p = self._init_model_p()
        self.gcn_c = self._init_model_c()
        self.gcn_v = self._init_model_v()

        self.lear1 = nn.Linear(self.emb_size * 4, self.emb_size)
        self.lear2 = nn.Linear(self.emb_size, self.emb_size)

        self.lear3 = nn.Linear(self.emb_size * 3, self.emb_size)
        self.lear4 = nn.Linear(self.emb_size, self.emb_size)
        self.w = torch.nn.Parameter(torch.FloatTensor([0.4, 0.3, 0.3]), requires_grad=True)
        self.weightu = torch.nn.Parameter(torch.FloatTensor(self.emb_size, self.emb_size))
        self.weighti = torch.nn.Parameter(torch.FloatTensor(self.emb_size * 3, self.emb_size))
        self.weightin = torch.nn.Parameter(torch.FloatTensor(self.emb_size * 3, self.emb_size))

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))

        # [n_users+n_items, n_users+n_items]
        self.sparse_norm_adj_p = self._convert_sp_mat_to_sp_tensor(self.adj_mat_p).to(self.device)
        self.sparse_norm_adj_c = self._convert_sp_mat_to_sp_tensor(self.adj_mat_c).to(self.device)
        self.sparse_norm_adj_v = self._convert_sp_mat_to_sp_tensor(self.adj_mat_v).to(self.device)

    def _init_model_p(self):
        return GraphConv(n_hops=self.context_hops,
                         n_users=self.n_users,
                         interact_mat=self.sparse_norm_adj_p,
                         edge_dropout_rate=self.edge_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _init_model_c(self):
        return GraphConv(n_hops=self.context_hops,
                         n_users=self.n_users,
                         interact_mat=self.sparse_norm_adj_c,
                         edge_dropout_rate=self.edge_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _init_model_v(self):
        return GraphConv(n_hops=self.context_hops,
                         n_users=self.n_users,
                         interact_mat=self.sparse_norm_adj_v,
                         edge_dropout_rate=self.edge_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, batch=None):
        user = batch['users']
        type_num = batch['type_n']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        pachas_uall, pachas_iall = self.gcn_p(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout)
        cart_uall, cart_iall = self.gcn_c(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout)
        view_uall, view_iall = self.gcn_v(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout)

        batch_size = len(user)

        pachas_u = pachas_uall[user]
        pachas_i_pos = pachas_iall[pos_item]
        pachas_i_neg = pachas_iall[neg_item[:, :self.K]]

        cart_u = cart_uall[user]
        cart_i_pos = cart_iall[pos_item]
        cart_i_neg = cart_iall[neg_item[:, :self.K]]

        view_u = view_uall[user]
        view_i_pos = view_iall[pos_item]
        view_i_neg = view_iall[neg_item[:, :self.K]]

        w0 = self.w[0]*type_num[0].unsqueeze(dim=1).unsqueeze(dim=1)
        w1 = self.w[1]*type_num[1].unsqueeze(dim=1).unsqueeze(dim=1)
        w2 = self.w[2]*type_num[2].unsqueeze(dim=1).unsqueeze(dim=1)

        w_0 = torch.exp(w0) / (torch.exp(w0)+torch.exp(w1)+torch.exp(w2))
        w_1 = torch.exp(w1) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2))
        w_2 = torch.exp(w2) / (torch.exp(w0) + torch.exp(w1) + torch.exp(w2))

        u_emb = w_0.mul(pachas_u) + w_1.mul(cart_u) + w_2.mul(view_u) #

        pos_i_emb = torch.cat((pachas_i_pos, cart_i_pos, view_i_pos), 2)  # view_i_pos
        neg_i_emb = torch.cat((pachas_i_neg, cart_i_neg, view_i_neg), 3)  # view_i_neg

        BPR_loss = self.create_bpr_loss(u_emb, pos_i_emb, neg_i_emb)

        u_p = self.pooling(pachas_u)
        u_c = self.pooling(cart_u)
        u_v = self.pooling(view_u)

        i_p = self.pooling(pachas_i_pos)
        i_c = self.pooling(cart_i_pos)
        i_v = self.pooling(view_i_pos)

        batch_size = u_p.size()[0]
        contr0 = InstanceLoss(self.tua0, batch_size)
        contr1 = InstanceLoss(self.tua1, batch_size)
        contr2 = InstanceLoss(self.tua2, batch_size)
        contr3 = InstanceLoss(self.tua3, batch_size)

        l0 = contr0.forward(u_p, u_v)
        l1 = contr1.forward(u_p, u_c)
        l2 = contr2.forward(i_p, i_c)
        l3 = contr3.forward(i_p, i_v)
        info_NCE = l0 + l1 + l2 + l3

        cluster0 = ClusterLoss(8, self.tua0)
        cluster1 = ClusterLoss(8, self.tua1)
        cluster2 = ClusterLoss(8, self.tua2)
        cluster3 = ClusterLoss(8, self.tua3)

        l_c0 = cluster0.forward(u_p, u_v)
        l_c1 = cluster1.forward(u_p, u_c)
        l_c2 = cluster2.forward(i_p, i_c)
        l_c3 = cluster3.forward(i_p, i_v)

        cluster_loss = l_c0 + l_c1 + l_c2 + l_c3
        loss = 1.0 * cluster_loss + 1.0 * info_NCE + BPR_loss
        return loss


    def pooling(self, embeddings):
        if self.pool == 'mean':
            return embeddings.mean(dim=1)
        elif self.pool == 'sum':
            return embeddings.sum(dim=1)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def generate(self):
        pachas_uall, pachas_iall = self.gcn_p(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout)
        cart_uall, cart_iall = self.gcn_c(self.user_embed,
                                          self.item_embed,
                                          edge_dropout=self.edge_dropout,
                                          mess_dropout=self.mess_dropout)
        view_uall, view_iall = self.gcn_v(self.user_embed,
                                          self.item_embed,
                                          edge_dropout=self.edge_dropout,
                                          mess_dropout=self.mess_dropout)

        user_purchase, item_purchase = self.pooling(pachas_uall), self.pooling(pachas_iall)
        user_cart, item_cart = self.pooling(cart_uall), self.pooling(cart_iall)
        user_view, item_view = self.pooling(view_uall), self.pooling(view_iall)
        user_gcn_emb = torch.cat((user_purchase, user_cart, user_view), 1)
        item_gcn_emb = torch.cat((item_purchase, item_cart, item_view), 1)
        return user_gcn_emb, item_gcn_emb

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, user_gcn_emb, pos_gcn_embs, neg_gcn_embs):
        batch_size = user_gcn_emb.shape[0]
        u_e = self.pooling(user_gcn_emb).mm(self.weightu)
        pos_e = self.pooling(pos_gcn_embs).mm(self.weighti)

        neg_e = self.pooling(neg_gcn_embs.view(-1, neg_gcn_embs.shape[2], neg_gcn_embs.shape[3])).view(batch_size, self.K, -1).matmul(self.weightin)

        pos_scores = torch.sum(torch.mul(u_e, pos_e), axis=1)
        neg_scores = torch.sum(torch.mul(u_e.unsqueeze(dim=1), neg_e), axis=-1)
        mf_loss = torch.mean(torch.log(1 + torch.exp(neg_scores - pos_scores.unsqueeze(dim=1)).sum(dim=1)))


        regularize = (torch.norm(user_gcn_emb[:, 0, :]) ** 2
                       + torch.norm(pos_gcn_embs[:, 0, :]) ** 2
                       + torch.norm(neg_gcn_embs[:, :, 0, :]) ** 2) / 2
        emb_loss = self.decay * regularize / batch_size

        return mf_loss + emb_loss
