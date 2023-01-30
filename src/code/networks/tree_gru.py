import torch
import torch.nn as nn
import torch.nn.functional as F

from multiprocessing import Pool


class TreeGRU(nn.Module):

    def __init__(self, op_dim, pred_dim , feature_dim, hidden_dim, hid_dim, embedding_type='tree_pool'):

        super(TreeGRU, self).__init__()
        self.op_dim = op_dim
        self.pred_dim = pred_dim
        self.feature_dim = feature_dim
        self.gru_hidden_dim = hidden_dim
        self.mlp_hid_dim = hid_dim

        self.operation_embed = nn.Linear(self.op_dim + self.feature_dim, self.mlp_hid_dim)
        self.predicate_embed = nn.Linear(self.pred_dim, self.mlp_hid_dim)
        self.sample_bitmap_embed = nn.Linear(1000, self.mlp_hid_dim)

        self.lstm_embed = nn.LSTM(pred_dim, hidden_dim, batch_first=True)

        if embedding_type == 'tree_pool':
            self.gru = nn.GRU(4 * self.mlp_hid_dim, self.gru_hidden_dim, batch_first=True)

        else:
            self.gru = nn.GRU(2 * self.gru_hidden_dim + 2 * self.mlp_hid_dim, self.gru_hidden_dim, batch_first=True)

        self.hid_mlp2_task1 = nn.Linear(self.gru_hidden_dim, self.mlp_hid_dim)
        self.hid_mlp2_task2 = nn.Linear(self.gru_hidden_dim, self.mlp_hid_dim)
        self.hid_mlp3_task1 = nn.Linear(self.mlp_hid_dim, self.mlp_hid_dim)
        self.hid_mlp3_task2 = nn.Linear(self.mlp_hid_dim, self.mlp_hid_dim)
        self.out_mlp2_task1 = nn.Linear(self.mlp_hid_dim, 1)
        self.out_mlp2_task2 = nn.Linear(self.mlp_hid_dim, 1)

        self.embedding_type = embedding_type

    def zeroes(self, dim):
        return torch.zeros(1, dim)

    def init_hidden(self, hidden_dim, batch_size=1):
        return (torch.zeros(1, batch_size, hidden_dim))
        

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

        if self.embedding_type == 'tree_pool':
            condition1_root = node.condition1_root
            condition2_root = node.condition2_root
        
        else:
            condition1_root = node.get_condition1_vector()
            condition2_root = node.get_condition2_vector()


        if self.embedding_type == 'tree_pool':
            if condition1_root is None:
                condition1_vector = self.zeroes(self.mlp_hid_dim)
            else:
                condition1_vector = self.pool(condition1_root)


            if condition2_root is None:
                condition2_vector = self.zeroes(self.mlp_hid_dim)
            else:
                condition2_vector = self.pool(condition2_root) 

        else:
            hidden1, cell1 = self.init_hidden(hidden_dim=self.gru_hidden_dim)
            _, (cond1_hid, cond1_cell) = self.lstm_embed(condition1_root, (hidden1, cell1))

            hidden2, cell2 = self.init_hidden(hidden_dim=self.gru_hidden_dim)
            _, (cond2_hid, cond2_cell) = self.lstm_embed(condition2_root, (hidden2, cell2))
                        
            condition1_vector = cond1_hid[0].view(1,-1)
            condition2_vector = cond2_hid[0].view(1,-1)

        operation_vector = self.operation_embed(torch.cat((node.get_torch_operation_vector(), node.get_torch_extra_info_vector()), 1))
        sample_bitmap_vector = self.sample_bitmap_embed(node.get_torch_sample_bitmap_vector()) * node.has_cond

        if len(node.children) == 0:
            left_hidden_state = self.init_hidden(self.gru_hidden_dim)
            right_hidden_state = self.init_hidden(self.gru_hidden_dim)

        elif len(node.children) == 1:
            _, left_hidden_state = self.tree_representation(node.children[0])
            right_hidden_state = self.init_hidden(self.gru_hidden_dim)

        else:
            # with Pool() as mp_pool:
            #     results = mp_pool.map(self.tree_representation, node.children)
            #     _, (left_hidden_state, left_cell_state) = results[0]
            #     _, (right_hidden_state, right_cell_state) = results[1]
            _, left_hidden_state = self.tree_representation(node.children[0])
            _, right_hidden_state = self.tree_representation(node.children[1])


        # print(operation_vector.shape)
        # print(condition1_vector.shape)
        # print(condition2_vector.shape)
        # print(sample_bitmap_vector.shape)
        input_vector = torch.cat((operation_vector, condition1_vector, condition2_vector, sample_bitmap_vector), 1)

        hidden_state = (left_hidden_state + right_hidden_state) / 2

        return self.gru(input_vector.view(1, 1, -1), hidden_state)


    def forward(self, node):
        _, hidden_state = self.tree_representation(node)

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