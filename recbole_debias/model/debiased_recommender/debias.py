# -*- coding: utf-8 -*-
# @Time   : 2023/4/27
# @Author : Haichao Zhang
# @Email  : Haichao.Zhang22@student.xjtlu.edu.cn

import torch
import torch.nn as nn

from recbole_debias.model.abstract_recommender import DebiasedRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss
from recbole.utils import InputType
import numpy as np
from torch.autograd import Function

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DEBIAS(DebiasedRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(DEBIAS, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.weight1 = 0.5
        self.weight2 = 2
        self.weight3 = 2
        # define layers and loss
        self.user_id_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.user_age_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.user_gender_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.user_occupation_embedding = nn.Embedding(self.n_users, self.embedding_size)

        #
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.item_interacion_num_level_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.item_gender_M_interacion_num_level_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.item_gender_F_interacion_num_level_embedding = nn.Embedding(self.n_items, self.embedding_size)

        #todo
        self.loss = BPRLoss()

        self.matching_network = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.conformity_network = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.item_network = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        self.domain_classfier = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        self.p = 0
        self.alpha = 2. / (1. + np.exp(-10 * self.p)) - 1
        # parameters initialization
        self.apply(xavier_normal_initialization)
    def get_user_embedding(self, interaction):
        id_embedding = self.user_id_embedding(interaction[self.USER_ID])
        # age_embedding = self.user_age_embedding(interaction["age_level"].to(torch.int64))
        # gender_embedding = self.user_gender_embedding(interaction["gender"])
        # occupation_embedding = self.user_occupation_embedding(interaction["occupation"])
        # return (id_embedding + age_embedding + gender_embedding + occupation_embedding) / 4
        return id_embedding
    def get_user_popular_embedding(self, interaction):
        r"""Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        popular_item0_embedding = self.item_id_embedding(interaction["item_interaction_popular0"])
        popular_item1_embedding = self.item_id_embedding(interaction["item_interaction_popular1"])
        popular_item2_embedding = self.item_id_embedding(interaction["item_interaction_popular2"])
        popular_item3_embedding = self.item_id_embedding(interaction["item_interaction_popular3"])
        popular_item4_embedding = self.item_id_embedding(interaction["item_interaction_popular4"])

        return self.get_user_embedding(interaction) + (popular_item0_embedding + popular_item1_embedding + popular_item2_embedding + popular_item3_embedding + popular_item4_embedding) / 5


    def get_user_unpopular_embedding(self, interaction):
        r"""Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        unpopular_item0_embedding = self.item_id_embedding(interaction["item_interaction_unpopular0"])
        unpopular_item1_embedding = self.item_id_embedding(interaction["item_interaction_unpopular1"])
        unpopular_item2_embedding = self.item_id_embedding(interaction["item_interaction_unpopular2"])
        unpopular_item3_embedding = self.item_id_embedding(interaction["item_interaction_unpopular3"])
        unpopular_item4_embedding = self.item_id_embedding(interaction["item_interaction_unpopular4"])

        return self.get_user_embedding(
            interaction) + (unpopular_item0_embedding + unpopular_item1_embedding + unpopular_item2_embedding + unpopular_item3_embedding + unpopular_item4_embedding) / 5


    def get_item_embedding(self, interaction):
        r"""Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        id_embedding = self.item_id_embedding(interaction[self.ITEM_ID])
        interacion_num_level_embedding = self.item_interacion_num_level_embedding(interaction["interaction_num_level"].to(torch.int64))
        # gender_M_interacion_num_level_embedding = self.item_gender_M_interacion_num_level_embedding(interaction["gender_M_interacion_num_level"].to(torch.int64))
        # gender_F_interacion_num_level_embedding = self.item_gender_F_interacion_num_level_embedding(interaction["gender_F_interacion_num_level"].to(torch.int64))
        # return (id_embedding + interacion_num_level_embedding) / 2
        return id_embedding

    def forward(self, interaction):
        dict = {}
        user_popular_embedding = self.get_user_popular_embedding(interaction)
        user_unpopular_embedding = self.get_user_unpopular_embedding(interaction)
        item_embedding = self.get_item_embedding(interaction)
        dict["user_popular_embedding"] = user_popular_embedding
        dict["user_unpopular_embedding"] = user_unpopular_embedding
        dict["item_embedding"] = item_embedding

        Ms = self.matching_network(user_popular_embedding)
        Mt = self.matching_network(user_unpopular_embedding)
        Cs = self.conformity_network(user_popular_embedding)
        Ct = self.conformity_network(user_unpopular_embedding)
        Iq = self.item_network(item_embedding)

        # save to dict
        dict["Ms"] = Ms
        dict["Mt"] = Mt
        dict["Cs"] = Cs
        dict["Ct"] = Ct
        dict["Iq"] = Iq

        # compute cos simliair  kl_div
        Ms_Mt_simliar = torch.cosine_similarity(Ms, Mt)
        Ms_Cs_simliar = torch.cosine_similarity(Ms, Cs)
        Ms_Ct_simliar = torch.cosine_similarity(Ms, Ct)
        Mt_Cs_simliar = torch.cosine_similarity(Mt, Cs)
        Mt_Ct_simliar = torch.cosine_similarity(Mt, Ct)
        # save to dict
        dict["Ms_Mt_simliar"] = Ms_Mt_simliar
        dict["Ms_Cs_simliar"] = Ms_Cs_simliar
        dict["Ms_Ct_simliar"] = Ms_Ct_simliar
        dict["Mt_Cs_simliar"] = Mt_Cs_simliar
        dict["Mt_Ct_simliar"] = Mt_Ct_simliar

        Yd = interaction["popular"].unsqueeze(-1)

        sigmod = nn.Sigmoid()

        Y1 = ((Iq * ((Yd * Ms) + (1 - Yd) * Mt))).sum(axis = 1)
        Y2 = ((Iq * ((Yd * Cs) + (1 - Yd) * Ct))).sum(axis = 1)
        Y3 = ((Iq * ((Yd * (Ms + Cs)) + (1 - Yd) * (Mt +Ct)))).sum(axis = 1)

        dict["Y1_predict"] = Y1
        Y1 = sigmod(Y1)
        Y2 = sigmod(Y2)
        Y3 = sigmod(Y3)

        dict["Y1"] = Y1
        dict["Y2"] = Y2
        dict["Y3"] = Y3


        if self.training:
            if self.p < 1:
                self.p += 1 / 15000
                self.alpha = 2. / (1. + np.exp(-10 * self.p)) - 1

        domin_network_Ms = self.domain_classfier(ReverseLayerF.apply(Ms, self.alpha))
        domin_network_Mt = self.domain_classfier(ReverseLayerF.apply(Mt, self.alpha))
        domin_network_Cs = self.domain_classfier(Cs)
        domin_network_Ct = self.domain_classfier(Ct)
        dict["domin_network_Ms"] = domin_network_Ms
        dict["domin_network_Mt"] = domin_network_Mt
        dict["domin_network_Cs"] = domin_network_Cs
        dict["domin_network_Ct"] = domin_network_Ct

        return dict

    def calculate_loss(self, interaction):
        dict = self.forward(interaction)

        bce_loss = nn.BCELoss()
        bce_loss1  = bce_loss(dict["Y1"], interaction["label"])
        bce_loss2 = bce_loss(dict["Y2"], interaction["label"])
        bce_loss3 = bce_loss(dict["Y3"], interaction["label"])

        gm = 0.5
        sim_loss1 = torch.exp(dict["Ms_Mt_simliar"] / gm)
        sim_loss1 = sim_loss1.sum(axis = 0) / len(interaction)
        sim_loss1 = -sim_loss1

        sim_loss2 = (torch.exp(dict["Ms_Cs_simliar"]) + torch.exp(dict["Ms_Ct_simliar"]) + torch.exp(dict["Mt_Cs_simliar"]) + torch.exp(dict["Mt_Ct_simliar"]))
        sim_loss2 = sim_loss2.sum(axis = 0) / len(interaction)


        causal_loss = torch.log(1 + torch.exp( (dict["Iq"] * dict["Ms"]).sum(axis=1) - (dict["Iq"] * dict["Mt"]).sum(axis=1)))
        causal_loss = causal_loss.sum(axis=0) / len(interaction)

        d0 = torch.zeros_like(dict["domin_network_Ms"])
        d1 = torch.ones_like(dict["domin_network_Ms"])

        domain_loss = bce_loss(dict["domin_network_Cs"], d1) + bce_loss(dict["domin_network_Ct"], d0) \
                      + bce_loss(dict["domin_network_Ms"], d1) + bce_loss(dict["domin_network_Mt"], d0)
        domain_loss = domain_loss / 4
        total_loss = (bce_loss1 + bce_loss2 + bce_loss3) / 10 + sim_loss1 * self.weight1 + sim_loss2 * self.weight1 + causal_loss * self.weight2\
                     + domain_loss * self.weight3
        # return total_loss
        return ((bce_loss1 + bce_loss2 + bce_loss3),   sim_loss2 * self.weight1, causal_loss * self.weight2, domain_loss * self.weight3)


    def predict(self, interaction):
        dict = self.forward(interaction)
        return dict["Y1_predict"]

    def full_sort_predict(self, interaction):
        user_popular_embedding = self.get_user_popular_embedding(interaction)
        user_unpopular_embedding = self.get_user_unpopular_embedding(interaction)
        all_item_e = self.item_id_embedding.weight

        Yd = interaction["popular"].unsqueeze(-1)

        Ms = self.matching_network(user_popular_embedding)
        Mt = self.matching_network(user_unpopular_embedding)

        Y1 = torch.matmul(((Yd * Ms) + (1 - Yd) * Mt), all_item_e.transpose(0, 1))  # [user_num,item_num]

        return Y1.view(-1)