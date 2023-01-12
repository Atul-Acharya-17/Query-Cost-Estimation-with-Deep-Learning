from ..models.TreePool import TreePool
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
from ..plan.map import physic_ops_id, compare_ops_id, bool_ops_id
from ..constants import DATA_ROOT
from ..plan.encode_plan import obtain_upper_bound_query_size
import numpy as np
import pickle

from ..plan.tree_vector import TreeVector, PlanNodeVector

def normalize_label(labels, mini, maxi):
    labels_norm = (labels - mini) / (maxi - mini)
    labels_norm = torch.min(labels_norm, torch.ones_like(labels_norm))
    labels_norm = torch.max(labels_norm, torch.zeros_like(labels_norm))
    return labels_norm

def unnormalize(vecs, mini, maxi):
    return (vecs * (maxi - mini) + mini)

def q_error(pred, target, mini, maxi):
    pred = unnormalize(pred, mini, maxi)
    target = unnormalize(target, mini, maxi)
    if pred == 0 and target == 0:
        return 1.0
    elif pred == 0:
        return target
    elif target == 0:
        return pred
    else:
        return max(pred, target) / min(pred, target)

def get_batch_job(batch_id, phase, directory):
    suffix = phase + "_"

    with open(f'{directory}/input_batch_{suffix+str(batch_id)}.pkl', 'rb') as handle:
        input_batch = pickle.load(handle)
    with open(f'{directory}/target_cost_{suffix+str(batch_id)}.pkl', 'rb') as handle:
        target_cost = pickle.load(handle)
    with open(f'{directory}/target_cardinality_{suffix+str(batch_id)}.pkl', 'rb') as handle:
        target_card = pickle.load(handle)

    return input_batch, target_cost, target_card

def train(train_start, train_end, validate_start, validate_end, num_epochs, directory):
    hidden_dim = 128
    hid_dim = 256
    model = TreePool(physic_op_total_num, bool_ops_total_num + compare_ops_total_num + column_total_num + max_string_dim, hidden_dim, hid_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    start = time.time()
    for epoch in range(num_epochs):
        cost_loss_total = 0.
        card_loss_total = 0.
        model.train()
        for batch_idx in range(train_start, train_end):
            input_batch, target_cost, target_cardinality = get_batch_job(batch_idx, phase='train', directory=directory)
            target_cost, target_cardinality= torch.FloatTensor(target_cost), torch.FloatTensor(target_cardinality)
            target_cost, target_cardinality = Variable(target_cost), Variable(target_cardinality)

            optimizer.zero_grad()

            for idx in range(len(input_batch)):
                required_input = input_batch[idx]
                cost = target_cost[idx]
                card = target_cardinality[idx]

                estimate_cost, estimate_cardinality = model(required_input)
                target_cost = target_cost
                target_cardinality = target_cardinality
            # print (card_loss.item(),card_loss_median.item(),card_loss_max.item(),card_max_idx.item())
            # print (cost_loss.item(),cost_loss_median.item(),cost_loss_max.item(),cost_max_idx.item())
                cost_loss = q_error(estimate_cost[0], cost, cost_label_min, cost_label_max)
                card_loss = q_error(estimate_cardinality[0], card, card_label_min, card_label_max)
                loss = cost_loss + card_loss
            loss.backward()
            optimizer.step()
            cost_loss_total += cost_loss.item()
            card_loss_total += card_loss.item()
            start = time.time()
            end = time.time()
        batch_num = train_end - train_start
        print("Epoch {}, training cost loss: {}, training card loss: {}".format(epoch, cost_loss_total/batch_num, card_loss_total/batch_num))

    end = time.time()
    print (end-start)

    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='random')
    parser.add_argument('--version', default='original')
    parser.add_argument('--name', default='base')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    dataset = args.dataset
    version = args.version
    name = args.name

    if dataset == 'census13':
        from ..data_preparation.census13 import columns_id, indexes_id, tables_id, max_string_dim

    plan_node_max_num, condition_max_num, cost_label_min, cost_label_max, card_label_min, card_label_max = obtain_upper_bound_query_size(str(DATA_ROOT) + "/" + dataset + "/workload/plans/" + "train_plans_encoded.json")

    index_total_num = len(indexes_id)
    table_total_num = len(tables_id)
    column_total_num = len(columns_id)
    physic_op_total_num = len(physic_ops_id)
    compare_ops_total_num = len(compare_ops_id)
    bool_ops_total_num = len(bool_ops_id)
    condition_op_dim = bool_ops_total_num + compare_ops_total_num + column_total_num
    condition_op_dim_pro = bool_ops_total_num + column_total_num + 3

    directory = str(DATA_ROOT) + "/" + dataset + "/workload/tree_data/"

    train(0, 15, 0, 0, 200, directory=directory)