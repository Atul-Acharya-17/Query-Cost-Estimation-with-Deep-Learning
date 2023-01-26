import argparse
import pickle
import numpy as np

import math
import json

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from ..plan.utils import unnormalize, class2json
from ..plan.entities import PredicateNodeVector, PlanNodeVector
from .loss_fn import q_error
from ..plan.map import physic_ops_id, compare_ops_id, bool_ops_id
from ..constants import DATA_ROOT, NUM_TRAIN, NUM_VAL, NUM_TEST, BATCH_SIZE
from ..plan.utils import obtain_upper_bound_query_size, obtain_upper_bound_query_size_intermediate
from ..networks.node_level_predictor import NodePredictor
from .helpers import get_batch_job_tree


def q_error_loss(pred, target, mini, maxi):
    preo = pred
    targeto = target
    pred = unnormalize(pred, mini=mini, maxi=maxi)
    target = unnormalize(target, mini=mini, maxi=maxi)
    q_err = q_error(pred, target)
    # if math.isnan(q_err):
    #     print(pred, target)
    #     print(preo, targeto)
    #     exit()
    return q_err

def modified_q_loss(pred, target, mini, maxi):
    pred = unnormalize(pred, mini=mini, maxi=maxi)
    target = unnormalize(target, mini=mini, maxi=maxi)

    if pred == 0 or target == 0:
        return abs(pred - target)
    else:
        return max(pred, target) / min(pred, target)

def train(train_start, train_end, validate_start, validate_end, num_epochs, directory):

    mlp_hid_dim = 256
    model = NodePredictor(physic_op_total_num, bool_ops_total_num + compare_ops_total_num + column_total_num + max_string_dim, feature_num, mlp_hid_dim, embedding_type=embedding_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        cost_loss_total = 0.
        card_loss_total = 0.
        num_samples = 0
        model.train()
        for batch_idx in range(train_start, train_end + 1):
            input_batch, target_cost, target_cardinality = get_batch_job_tree(batch_idx, phase='train', directory=directory)
            target_cost, target_cardinality= torch.FloatTensor(target_cost), torch.FloatTensor(target_cardinality)
            target_cost, target_cardinality = Variable(target_cost), Variable(target_cardinality)

            optimizer.zero_grad()

            num_samples += len(input_batch)

            loss = 0.0

            num_items = 0

            for idx in range(len(input_batch)):
                plan = input_batch[idx]
                cost = target_cost[idx]
                card = target_cardinality[idx]

                estimate_cost, estimate_cardinality, cost_list, card_list = model(plan)

                cost_loss = q_error_loss(estimate_cost, cost, cost_label_min, cost_label_max)
                card_loss = q_error_loss(estimate_cardinality, card, card_label_min, card_label_max)
                

                if torch.is_tensor(cost_loss):
                    cost_loss = cost_loss.item()
                
                if torch.is_tensor(card_loss): 
                    card_loss = card_loss.item()

                cost_loss_total += cost_loss
                card_loss_total += card_loss

                cost_losses = [q_error_loss(pred, actual, cost_label_min, cost_label_max) for (pred, actual) in cost_list]
                card_losses = [q_error_loss(pred, actual, card_label_min, card_label_max) for (pred, actual) in card_list]

                loss += sum(cost_losses) + sum(card_losses)

                num_items += len(cost_list)
                # print(f"{len(cost_list)} , {len(card_list)}")
            loss /= num_items

            # print(num_items)

            # loss += 0.01 * sum([torch.norm(weight) for weight in model.parameters()])
            loss.backward()
            optimizer.step()

        print("Epoch {}, training cost loss: {}, training card loss: {}".format(epoch, cost_loss_total/num_samples, card_loss_total/num_samples))

        valid_cost_losses, valid_card_losses = validate(model, validate_start, validate_end, directory, phase='valid')
        avg_cost_loss = sum(valid_cost_losses) / len(valid_cost_losses)
        avg_card_loss = sum(valid_card_losses) / len(valid_card_losses)

        print("Epoch {}, validation cost loss: {}, validation card loss: {}".format(epoch, avg_cost_loss, avg_card_loss))

    return model

def validate(model, start_idx, end_idx, directory, phase='valid'):
    model.eval()

    cost_losses = []
    card_losses = []

    for batch_idx in range(start_idx, end_idx + 1):
        input_batch, target_cost, target_cardinality = get_batch_job_tree(batch_idx, phase=phase, directory=directory)
        target_cost, target_cardinality= torch.FloatTensor(target_cost), torch.FloatTensor(target_cardinality)
        target_cost, target_cardinality = Variable(target_cost), Variable(target_cardinality)
        for idx in range(len(input_batch)):
            plan = input_batch[idx]
            cost = target_cost[idx]
            card = target_cardinality[idx]

            estimate_cost, estimate_cardinality, _, _ = model(plan, phase='inference')
            target_cost = target_cost
            target_cardinality = target_cardinality

            cost_loss = q_error_loss(estimate_cost, cost, cost_label_min, cost_label_max)
            card_loss = q_error_loss(estimate_cardinality, card, card_label_min, card_label_max)
            
            if torch.is_tensor(cost_loss):
                cost_loss = cost_loss.item()
            if torch.is_tensor(card_loss):
                card_loss = card_loss.item()

            cost_losses.append(cost_loss)
            card_losses.append(card_loss)

    return cost_losses, card_losses


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='random')
    parser.add_argument('--version', default='original')
    parser.add_argument('--name', default='base')
    parser.add_argument('--embedding-type', default='tree_pool')
    parser.add_argument('--intermediate', default='False')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    dataset = args.dataset
    version = args.version
    name = args.name

    intermediate = args.intermediate

    embedding_type = args.embedding_type

    if embedding_type not in ['tree_pool', 'lstm']:
        print('Invalid embedding type')
        raise

    if dataset == 'census13':
        from ..dataset.census13 import columns_id, indexes_id, tables_id, max_string_dim

    elif dataset == 'forest10':
        from ..dataset.forest10 import columns_id, indexes_id, tables_id, max_string_dim

    if dataset == 'power7':
        from ..dataset.power7 import columns_id, indexes_id, tables_id, max_string_dim

    elif dataset == 'dmv11':
        from ..dataset.dmv11 import columns_id, indexes_id, tables_id, max_string_dim

    if intermediate == 'True':
        plan_node_max_num, condition_max_num, cost_label_min, cost_label_max, card_label_min, card_label_max = obtain_upper_bound_query_size_intermediate(str(DATA_ROOT) + "/" + dataset + "/workload/plans/" + "train_plans_encoded.json")
    else:
        plan_node_max_num, condition_max_num, cost_label_min, cost_label_max, card_label_min, card_label_max = obtain_upper_bound_query_size(str(DATA_ROOT) + "/" + dataset + "/workload/plans/" + "train_plans_encoded.json")

    print(card_label_min, card_label_max)

    index_total_num = len(indexes_id)
    table_total_num = len(tables_id)
    column_total_num = len(columns_id)
    physic_op_total_num = len(physic_ops_id)
    compare_ops_total_num = len(compare_ops_id)
    bool_ops_total_num = len(bool_ops_id)
    condition_op_dim = bool_ops_total_num + compare_ops_total_num + column_total_num + max_string_dim
    feature_num = max(column_total_num, table_total_num, index_total_num)

    directory = str(DATA_ROOT) + "/" + dataset + "/workload/tree_data/"

    train_end = NUM_TRAIN // BATCH_SIZE - 1 if NUM_TRAIN % BATCH_SIZE == 0 else NUM_TRAIN // BATCH_SIZE
    valid_end = NUM_VAL // BATCH_SIZE - 1 if NUM_VAL % BATCH_SIZE == 0 else NUM_VAL // BATCH_SIZE
    test_end = NUM_TEST // BATCH_SIZE - 1 if NUM_TEST % BATCH_SIZE == 0 else NUM_VAL // BATCH_SIZE
    
    epochs = 200

    model = train(0, train_end, 0, valid_end, epochs, directory=directory)
    # cost_loss, card_loss = validate(model, 0, train_end, directory, 'train')
    # print(f"train cost loss: {cost_loss}, train cardinality loss: {card_loss}")
    # cost_loss, card_loss = validate(model, 0, valid_end, directory, 'valid')
    # print(f"valid cost loss: {cost_loss}, valid cardinality loss: {card_loss}")

    cost_losses, card_losses = validate(model, 0, test_end, directory, 'test')

    cost_metrics = {
        'max': np.max(cost_losses),
        '99th': np.percentile(cost_losses, 99),
        '95th': np.percentile(cost_losses, 95),
        '90th': np.percentile(cost_losses, 90),
        'median': np.median(cost_losses),
        'mean': np.mean(cost_losses),
    }

    card_metrics = {
        'max': np.max(card_losses),
        '99th': np.percentile(card_losses, 99),
        '95th': np.percentile(card_losses, 95),
        '90th': np.percentile(card_losses, 90),
        'median': np.median(card_losses),
        'mean': np.mean(card_losses),
    }


    print(f"cost metrics: {cost_metrics}, \ncardinality metrics: {card_metrics}")