import torch
import torch.nn as nn
import torch.nn.functional as F


class TreeNNBatch(nn.Module):

    def __init__(self, op_dim, pred_dim , feature_dim, hidden_dim, hid_dim, embedding_type='tree_pool'):

        super(TreeNNBatch, self).__init__()
        torch.manual_seed(0)
        self.op_dim = op_dim
        self.pred_dim = pred_dim
        self.feature_dim = feature_dim
        self.lstm_hidden_dim = hidden_dim
        self.mlp_hid_dim = hid_dim

        self.operation_embed = nn.Linear(self.op_dim, self.mlp_hid_dim)
        self.predicate_embed = nn.Linear(self.pred_dim, self.mlp_hid_dim)
        self.sample_bitmap_embed = nn.Linear(1000, self.mlp_hid_dim)
        self.feature_embed = nn.Linear(feature_dim, self.mlp_hid_dim)

        self.lstm_embed = nn.LSTM(pred_dim, hidden_dim, batch_first=True)
        self.representation = nn.Sequential(
            nn.Linear(5 * self.mlp_hid_dim + 6 * 128, 512), # 6 * 128 because size of left and right representation is 128, and we have left.left, left.right, right.left and right.right
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.hid_mlp2_task1 = nn.Linear(128, self.mlp_hid_dim)
        self.hid_mlp2_task2 = nn.Linear(128, self.mlp_hid_dim)
        self.hid_mlp3_task1 = nn.Linear(self.mlp_hid_dim, self.mlp_hid_dim)
        self.hid_mlp3_task2 = nn.Linear(self.mlp_hid_dim, self.mlp_hid_dim)
        self.out_mlp2_task1 = nn.Linear(self.mlp_hid_dim, 1)
        self.out_mlp2_task2 = nn.Linear(self.mlp_hid_dim, 1)

        self.embedding_type = embedding_type

    def zeroes(self, dim, batch_size=1):
        return torch.zeros(batch_size, dim)

    def init_hidden(self, hidden_dim, batch_size=1):
        return (torch.zeros(1, batch_size, hidden_dim),
                torch.zeros(1, batch_size, hidden_dim))
        

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


    def tree_representation(self, nodes):

        batch_size = len(nodes)

        condition1_vecs = []
        condition2_vecs = []

        operation_vecs = []
        sample_bitmap_vecs = []
        feature_vecs = []

        has_cond_vecs = []

        left_rep_vector = self.zeroes(dim=128, batch_size=batch_size)
        right_rep_vector = self.zeroes(dim=128, batch_size=batch_size)

        left_left = self.zeroes(dim=128, batch_size=batch_size)
        left_right = self.zeroes(dim=128, batch_size=batch_size)
        right_left = self.zeroes(dim=128, batch_size=batch_size)
        right_right = self.zeroes(dim=128, batch_size=batch_size)

        left_mask = []
        right_mask = []

        left_nodes = []
        right_nodes = []

        for idx, node in enumerate(nodes):
            condition1_root = node.condition1_root
            condition2_root = node.condition2_root

            if condition1_root is None:
                condition1_vector = self.zeroes(self.mlp_hid_dim, batch_size=1)
            else:
                condition1_vector = self.pool(condition1_root)

            if condition2_root is None:
                condition2_vector = self.zeroes(self.mlp_hid_dim, batch_size=1)

            else:
                condition2_vector = self.pool(condition2_root)

            condition1_vecs.append(condition1_vector)
            condition2_vecs.append(condition2_vector)

            operation_vecs.append(node.get_torch_operation_vector())
            sample_bitmap_vecs.append(node.get_torch_sample_bitmap_vector())
            feature_vecs.append(node.get_torch_extra_info_vector())

            has_cond_vecs.append(node.has_cond)

            if len(node.children) == 0:
                right_mask.append(0)
                left_mask.append(0)

            elif len(node.children) == 1:
                left_mask.append(1)
                right_mask.append(0)
                left_nodes.append(node.children[0])

            else:
                left_mask.append(1)
                right_mask.append(1)
                left_nodes.append(node.children[0])
                right_nodes.append(node.children[1])

        operation_vecs = torch.cat(operation_vecs, dim=0)
        sample_bitmap_vecs = torch.cat(sample_bitmap_vecs, dim=0)
        feature_vecs = torch.cat(feature_vecs, dim=0)

        has_cond_vecs = torch.unsqueeze(torch.Tensor(has_cond_vecs), 1).view(batch_size)

        condition1_vecs = torch.cat(condition1_vecs, dim=0)
        condition2_vecs = torch.cat(condition2_vecs, dim=0)

        operation_vector = self.operation_embed(operation_vecs)
        feature_vector = self.feature_embed(feature_vecs)

        sample_bitmap_vector = self.sample_bitmap_embed(sample_bitmap_vecs) * has_cond_vecs[:, None]

        if len(left_nodes) > 0:
            left_mask = torch.Tensor(left_mask) > 0
            left_reps, left_left_rep, left_right_rep = self.tree_representation(left_nodes)
            left_rep_vector[left_mask] = left_reps
            left_left[left_mask] = left_left_rep
            left_right[left_mask] = left_right_rep

        if len(right_nodes) > 0:
            right_mask = torch.Tensor(right_mask) > 0
            right_reps, right_left_rep, right_right_rep = self.tree_representation(right_nodes)
            right_rep_vector[right_mask] = right_reps
            right_left[right_mask] = right_left_rep
            right_right[right_mask] = right_right_rep

        
        input_vector = torch.cat((operation_vector, feature_vector, condition1_vecs, condition2_vecs, sample_bitmap_vector, left_rep_vector, right_rep_vector, left_left, left_right, right_left, right_right), 1)

        return self.representation(input_vector), left_rep_vector, right_rep_vector


    def forward(self, nodes, batch=True):
        output = self.tree_representation(nodes)

        cost = F.relu(self.hid_mlp2_task1(output))
        cost = F.relu(self.hid_mlp3_task1(cost))
        cost = self.out_mlp2_task1(cost)
        cost = F.sigmoid(cost)
        
        card = F.relu(self.hid_mlp2_task2(output))
        card = F.relu(self.hid_mlp3_task2(card))
        card = self.out_mlp2_task2(card)
        card = F.sigmoid(card)

        return cost, card
