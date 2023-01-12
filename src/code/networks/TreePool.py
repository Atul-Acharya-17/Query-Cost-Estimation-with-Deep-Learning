
import torch
import torch.nn as nn
import time
import torch.nn.functional as F

from multiprocessing import Pool


class TreePool(nn.Module):
    def __init__(self, op_dim, pred_dim , hidden_dim, hid_dim):
        super(TreePool, self).__init__()
        print(pred_dim)
        self.op_dim = op_dim
        self.pred_dim = pred_dim
        self.lstm_hidden_dim = hidden_dim
        self.mlp_hid_dim = hid_dim

        self.operation_embed = nn.Linear(self.op_dim, self.op_dim)
        self.predicate_embed = nn.Linear(self.pred_dim, self.pred_dim)
        self.sample_bitmap_embed = nn.Linear(1000, self.mlp_hid_dim)

        self.lstm = nn.LSTM(self.op_dim + 2 * self.pred_dim + self.mlp_hid_dim, self.lstm_hidden_dim, batch_first=True)

        self.hid_mlp2_task1 = nn.Linear(self.lstm_hidden_dim, self.mlp_hid_dim)
        self.hid_mlp2_task2 = nn.Linear(self.lstm_hidden_dim, self.mlp_hid_dim)
        self.hid_mlp3_task1 = nn.Linear(self.mlp_hid_dim, self.mlp_hid_dim)
        self.hid_mlp3_task2 = nn.Linear(self.mlp_hid_dim, self.mlp_hid_dim)
        self.out_mlp2_task1 = nn.Linear(self.mlp_hid_dim, 1)
        self.out_mlp2_task2 = nn.Linear(self.mlp_hid_dim, 1)

    def zeroes(self, dim):
        return torch.zeros(1, dim)

    def init_hidden(self, hidden_dim, batch_size=1):
        return (torch.zeros(1, 1, hidden_dim),
                torch.zeros(1, 1, hidden_dim))
        

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

    
    def tree_representation(self, node):

        condition1_root = node.condition1_root

        if condition1_root is None:
            condition1_vector = self.zeroes(self.pred_dim)

        else:
            condition1_vector = self.pool(condition1_root)

        condition2_root = node.condition2_root

        if condition2_root is None:
            condition2_vector = self.zeroes(self.pred_dim)
        else:
            condition2_vector = self.pool(condition2_root) 

        #operation_vector = self.operation_embed(node.get_torch_operation_vector())
        operation_vector = node.get_torch_operation_vector()
        sample_bitmap_vector = self.sample_bitmap_embed(node.get_torch_sample_bitmap_vector()) * node.has_cond

        if len(node.children) == 0:
            left_hidden_state, left_cell_state = self.init_hidden(self.lstm_hidden_dim)
            right_hidden_state, right_cell_state = self.init_hidden(self.lstm_hidden_dim)

        elif len(node.children) == 1:
            _, (left_hidden_state, left_cell_state) = self.tree_representation(node.children[0])
            right_hidden_state, right_cell_state = self.init_hidden(self.lstm_hidden_dim)

        else:
            # with Pool() as mp_pool:
            #     results = mp_pool.map(self.tree_representation, node.children)
            #     _, (left_hidden_state, left_cell_state) = results[0]
            #     _, (right_hidden_state, right_cell_state) = results[1]
            _, (left_hidden_state, left_cell_state) = self.tree_representation(node.children[0])
            _, (right_hidden_state, right_cell_state) = self.tree_representation(node.children[1])


        input_vector =  torch.cat((operation_vector, condition1_vector, condition2_vector, sample_bitmap_vector), 1)

        hidden_state = (left_hidden_state + right_hidden_state) / 2

        cell_state = (left_cell_state + right_cell_state) / 2
        return self.lstm(input_vector.view(1, 1, -1), (hidden_state, cell_state))


    def forward(self, node):
        _, (hidden_state, cell_state) = self.tree_representation(node)

        output = hidden_state[0].view(1, -1)

        cost = F.relu(self.hid_mlp2_task1(output))
        cost = F.relu(self.hid_mlp3_task1(cost))
        cost = self.out_mlp2_task1(cost)
        cost = F.sigmoid(cost)
        
        card = F.relu(self.hid_mlp2_task2(output))
        card = F.relu(self.hid_mlp3_task2(card))
        card = self.out_mlp2_task2(card)
        card = F.sigmoid(card)

        return cost, card

        
        