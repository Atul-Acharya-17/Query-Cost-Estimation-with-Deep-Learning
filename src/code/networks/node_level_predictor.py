import torch
import torch.nn as nn
import torch.nn.functional as F

from multiprocessing import Pool

import math


class NodePredictor(nn.Module):

    def __init__(self, op_dim, pred_dim , feature_dim, hid_dim, embedding_type='tree_pool'):

        super(NodePredictor, self).__init__()
        torch.manual_seed(0)
        self.op_dim = op_dim
        self.pred_dim = pred_dim
        self.feature_dim = feature_dim
        self.mlp_hid_dim = hid_dim

        self.operation_embed = nn.Linear(self.op_dim + self.feature_dim, self.mlp_hid_dim)
        self.predicate_embed = nn.Linear(self.pred_dim, self.mlp_hid_dim)
        self.sample_bitmap_embed = nn.Linear(1000, self.mlp_hid_dim)

        self.cost_card_embed = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU()
        )
        
        self.node_embed = nn.Linear(4 * self.mlp_hid_dim, 256)

        self.middle = nn.Sequential(
                nn.Linear(256 + 2 * 64, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU()
            )

        self.cost_task = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        self.card_task = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        self.embedding_type = embedding_type

    def zeroes(self, dim):
        return torch.zeros(1, dim)

    def ones(self, dim):
        return torch.ones(1, dim)

    def pool(self, condition_root):
        if condition_root.op_type == "Bool":
            vectors = [self.pool(child) for child in condition_root.children]

            result_vector = None

            for vec in vectors:
                if result_vector is None:
                    result_vector = vec
                else:
                    result_vector = torch.min(result_vector, vec) if condition_root.operator == "AND" else torch.max(result_vector, vec)

            return result_vector

        if condition_root.op_type == "Compare":
            return self.predicate_embed(condition_root.get_torch_tensor())


    def predict_node(self, node, phase='train'):
        condition1_root = node.condition1_root
        condition2_root = node.condition2_root

        if condition1_root is None:
            condition1_vector = self.zeroes(self.mlp_hid_dim)
        else:
            condition1_vector = self.pool(condition1_root)

        if condition2_root is None:
            condition2_vector = self.zeroes(self.mlp_hid_dim)
        else:
            condition2_vector = self.pool(condition2_root) 


        operation_vector = self.operation_embed(torch.cat((node.get_torch_operation_vector(), node.get_torch_extra_info_vector()), 1))
        sample_bitmap_vector = self.sample_bitmap_embed(node.get_torch_sample_bitmap_vector()) * node.has_cond

        if len(node.children) == 0:
            # this means Scan operation
            left_child_vector = self.ones(2)
            right_child_vector = self.ones(2)

        elif len(node.children) == 1:
            left_child_vector = self.predict_node(node.children[0])

            if phase == 'train':
                left_child_vector = torch.FloatTensor([[node.children[0].cost, node.children[0].cardinality]])
            right_child_vector = self.zeroes(2)

        else:
            # with Pool() as mp_pool:
            #     results = mp_pool.map(self.tree_representation, node.children)
            #     _, (left_hidden_state, left_cell_state) = results[0]
            #     _, (right_hidden_state, right_cell_state) = results[1]

            left_child_vector = self.predict_node(node.children[0])
            right_child_vector = self.predict_node(node.children[1])

            if phase == 'train':
                left_child_vector = torch.FloatTensor([[node.children[0].cost, node.children[0].cardinality]])
                right_child_vector = torch.FloatTensor([[node.children[1].cost, node.children[1].cardinality]])


        input_vector = torch.cat((operation_vector, condition1_vector, condition2_vector, sample_bitmap_vector), 1)

        node_embedding = F.relu(self.node_embed(input_vector))

        cost_card_left = self.cost_card_embed(left_child_vector)
        cost_card_right = self.cost_card_embed(right_child_vector)

        #prediction = F.relu(self.fc1(torch.cat((node_embedding, cost_card_left, cost_card_right), 1)))
        middle_out = self.middle(torch.cat((node_embedding, cost_card_left, cost_card_right), 1))

        cost = self.cost_task(middle_out)
        card = self.card_task(middle_out)

        self.card_global_accumulator.append((card[0], node.cardinality))
        self.cost_global_accumulator.append((cost[0], node.cost))

        return torch.cat((cost, card), 1)


    def forward(self, node, phase='train'):

        assert phase in ['train', 'inference']

        self.cost_global_accumulator = []
        self.card_global_accumulator = []

        output = self.predict_node(node, phase=phase)

        if math.isnan(output[0][0]) or math.isnan(output[0][1]):
            for param in self.parameters():
                print(param)
            exit()

        return output[0][0], output[0][1], self.cost_global_accumulator, self.card_global_accumulator 
        