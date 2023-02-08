import argparse
import pandas as pd
import numpy as np
import time
import json
import math
import torch

import xgboost

from ..plan.utils import unnormalize, unnormalize_log
from ..plan.entities import PredicateNodeVector, PlanNodeVector
from .loss_fn import q_error
from ..plan.utils import obtain_upper_bound_query_size, obtain_upper_bound_query_size_log
from .helpers import get_batch_job_tree
from ..plan.map import physic_ops_id, compare_ops_id, bool_ops_id

from ..networks.tree_lstm import TreeLSTMBatch
from ..networks.tree_xgb import TreeXGB

from ..constants import DATA_ROOT, RESULT_ROOT, NUM_TRAIN, NUM_VAL, NUM_TEST, BATCH_SIZE, JOB_LIGHT, JOB_TRAIN, SCALE, SYNTHETIC


def q_error_loss(pred, target, mini, maxi):

    if dataset == 'imdb':
        pred = unnormalize_log(pred, mini=mini, maxi=maxi)
        target = unnormalize_log(target, mini=mini, maxi=maxi)    
    else:    
        pred = unnormalize(pred, mini=mini, maxi=maxi)
        target = unnormalize(target, mini=mini, maxi=maxi)

    q_err = q_error(pred, target)
    print(pred, target, q_err)
    return q_err


def q_loss(pred, target, mini, maxi):
    pred = unnormalize_log(pred, mini=mini, maxi=maxi)
    q_err = q_error(pred, target)
    print(pred, target, q_err)
    return q_err



def flatten_plan(plan, tree_pooler):
    op_type = np.array(plan.operator_vec)
    feature = np.array(plan.extra_info_vec)
    bitmap = np.array(plan.sample_vec) * plan.has_cond

    cond1 = plan.condition1_root
    cond2 = plan.condition2_root

    if cond1 is None:
        condition1_vector = np.zeros(256)
    else:
        condition1_vector = tree_pooler(cond1)[0]
        condition1_vector = condition1_vector.cpu().detach().numpy()

    if cond2 is None:
        condition2_vector = np.zeros(256)
    else:
        condition2_vector = tree_pooler(cond2)[0]
        condition2_vector = condition2_vector.cpu().detach().numpy()    

    cost = np.array(plan.cost, dtype="float64")
    card = np.array(plan.cardinality, dtype="float64")

    right_card = np.array([1])
    right_cost = np.array([0])
    left_card = np.array([1])
    left_cost = np.array([0])

    if len(plan.children) == 1: #  Only left child
        x_data, y_cost, y_card, left_cost, left_card = flatten_plan(plan.children[0], tree_pooler)

    elif len(plan.children) == 2: # 2 children
        x_data_left, y_cost_left, y_card_left, left_cost, left_card = flatten_plan(plan.children[0], tree_pooler)
        x_data_right, y_cost_right, y_card_right, right_cost, right_card = flatten_plan(plan.children[1], tree_pooler)

        x_data = x_data_left + x_data_right
        y_cost = y_cost_left + y_cost_right
        y_card = y_card_left + y_card_right

    else:
        x_data = []
        y_cost = []
        y_card = []

    data = np.concatenate((op_type, feature, bitmap, condition1_vector, condition2_vector, left_cost, right_cost, left_card, right_card))
    data = data.reshape((1, -1))
    x_data.append(data)
    y_cost.append(cost)
    y_card.append(card)

    return x_data, y_cost, y_card, cost, card



def generate_training_data(train_start, train_end, directory, phase, tree_pooler):
    X_train = []
    y_cost = []
    y_card = []
    for batch_idx in range(train_start, train_end + 1):
        print(f'{batch_idx + 1}/{train_end}', end='\r')
        input_batch, target_cost, target_cardinality = get_batch_job_tree(batch_idx, phase=phase, directory=directory)
        for plan in input_batch:
            x, cost, card, _, _ = flatten_plan(plan, tree_pooler)
            X_train += x
            y_cost += cost
            y_card += card

    return X_train, y_cost, y_card


def train_xgb(train_start, train_end, directory, phase, model):
    X_train, y_cost, y_card = generate_training_data(train_start, train_end, directory=directory, phase=phase, tree_pooler=model.pool)

    X_train = np.concatenate(X_train, axis=0)
    y_cost = np.concatenate(y_cost, axis=0).reshape(-1, 1)
    y_card = np.concatenate(y_card, axis=0).reshape(-1, 1)

    print(X_train.shape)
    # print(y_cost.shape)
    # print(y_card.shape)

    xgb_cost = xgboost.XGBRegressor(n_estimators=100, max_depth=8, eta=0.1, subsample=0.7, colsample_bytree=0.8, seed=0)
    xgb_cost.fit(X_train, y_cost)

    xgb_card = xgboost.XGBRegressor(n_estimators=100, max_depth=8, eta=0.1, subsample=0.7, colsample_bytree=0.8, seed=0)
    xgb_card.fit(X_train, y_card)

    return xgb_cost, xgb_card 


def evaluate_xgb(xgb, start_idx, end_idx, directory, phase):
    cost_losses = []
    card_losses = []
    
    inference_times = []

    for batch_idx in range(start_idx, end_idx + 1):
        input_batch, target_cost, target_cardinality, true_cost, true_card = get_batch_job_tree(batch_idx, phase=phase, directory=directory, get_unnorm=True)

        for idx in range(len(input_batch)):
            plan = input_batch[idx]
            cost = target_cost[idx].item()
            card = target_cardinality[idx].item()

            real_cost = true_cost[idx].item()
            real_card = true_card[idx].item()

            start_time = time.time()
            estimated_cost, estimated_card = xgb.predict(plan)
            end_time = time.time()

            cost_loss = q_loss(estimated_cost[0], real_cost, cost_label_min, cost_label_max)
            card_loss = q_loss(estimated_card[0], real_card, card_label_min, card_label_max)

            cost_losses.append(cost_loss)
            card_losses.append(card_loss)
                
            inference_times.append(end_time - start_time)

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
    
    time_metrics = {
        'max': np.max(inference_times),
        '99th': np.percentile(inference_times, 99),
        '95th': np.percentile(inference_times, 95),
        '90th': np.percentile(inference_times, 90),
        'median': np.median(inference_times),
        'mean': np.mean(inference_times),        
    }


    print(f"cost metrics: {cost_metrics}, \ncardinality metrics: {card_metrics}, \nInference Time metrics: {time_metrics}")
    
    stats_df = pd.DataFrame(list(zip(cost_losses, card_losses, inference_times)), columns=['cost_errors', 'card_errors', 'inference_time'])
    stats_df.to_csv(str(RESULT_ROOT) + "/output/" + dataset + f"/results_xgb_{phase}_new.csv")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='imdb')
    parser.add_argument('--name', default='tree_xgb')
    parser.add_argument('--embedding-type', default='tree_pool')
    parser.add_argument('--size', default=10000, type=int)
    parser.add_argument('--method', default='tree_xgb')
    parser.add_argument('--num-models', default=5, type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    
    dataset = args.dataset
    size = args.size
    num_models = args.num_models

    train_end = NUM_TRAIN // BATCH_SIZE - 1 if NUM_TRAIN % BATCH_SIZE == 0 else NUM_TRAIN // BATCH_SIZE
    valid_end = NUM_VAL // BATCH_SIZE - 1 if NUM_VAL % BATCH_SIZE == 0 else NUM_VAL // BATCH_SIZE
    test_end = NUM_TEST // BATCH_SIZE - 1 if NUM_TEST % BATCH_SIZE == 0 else NUM_VAL // BATCH_SIZE

    job_train_end = JOB_TRAIN // BATCH_SIZE - 1 if JOB_TRAIN % BATCH_SIZE == 0 else JOB_TRAIN // BATCH_SIZE
    job_light_end = JOB_LIGHT // BATCH_SIZE - 1 if JOB_LIGHT % BATCH_SIZE == 0 else JOB_LIGHT // BATCH_SIZE
    scale_end = SCALE // BATCH_SIZE - 1 if SCALE % BATCH_SIZE == 0 else SCALE // BATCH_SIZE
    synthetic_end = SYNTHETIC // BATCH_SIZE - 1 if SYNTHETIC % BATCH_SIZE == 0 else SYNTHETIC // BATCH_SIZE

    if dataset == 'census13':
        from ..dataset.census13 import columns_id, indexes_id, tables_id, max_string_dim

    elif dataset == 'forest10':
        from ..dataset.forest10 import columns_id, indexes_id, tables_id, max_string_dim

    if dataset == 'power7':
        from ..dataset.power7 import columns_id, indexes_id, tables_id, max_string_dim

    elif dataset == 'dmv11':
        from ..dataset.dmv11 import columns_id, indexes_id, tables_id, max_string_dim

    elif dataset == 'imdb':
        from ..dataset.imdb import columns_id, indexes_id, tables_id, max_string_dim

    train_path = "train_plan_100000"

    plan_node_max_num, condition_max_num, cost_label_min, cost_label_max, card_label_min, card_label_max = obtain_upper_bound_query_size_log(str(DATA_ROOT) + "/" + dataset + "/workload/plans/" + f"{train_path}_encoded.json")

    print(card_label_min, card_label_max)
    print(cost_label_min, cost_label_max)
    

    index_total_num = len(indexes_id)
    table_total_num = len(tables_id)
    column_total_num = len(columns_id)
    physic_op_total_num = len(physic_ops_id)
    compare_ops_total_num = len(compare_ops_id)
    bool_ops_total_num = len(bool_ops_id)
    condition_op_dim = bool_ops_total_num + compare_ops_total_num + column_total_num + max_string_dim
    feature_num = column_total_num + table_total_num + index_total_num + 1


    hidden_dim = 128
    mlp_hid_dim = 256

    model = TreeLSTMBatch(physic_op_total_num, bool_ops_total_num + compare_ops_total_num + column_total_num + max_string_dim, feature_num, hidden_dim, mlp_hid_dim, embedding_type='tree_pool')
    pool_path = str(RESULT_ROOT) + '/models/' + dataset + '/tree_lstm_100000.pt'
 
    model.load_state_dict(torch.load(pool_path))

    model.eval()

    directory = str(DATA_ROOT) + "/" + dataset + "/workload/tree_data/"

    phase=f'train_plan_{size}'

    xgb = TreeXGB(tree_pooler=model.pool)
    train_size = size // BATCH_SIZE - 1 if size % BATCH_SIZE == 0 else size // BATCH_SIZE

    for i in range(num_models):
        start_time = time.time()
        xgb_cost, xgb_card = train_xgb(i*(int(train_size/num_models)), (i+1)*int(train_size/num_models), directory, phase, model)
        end_time = time.time()
        xgb.add_estimators(xgb_cost, xgb_card)
        print(f'Time to train model {i+1}: {end_time-start_time}')

    ends = {
        "train": train_end,
        "valid": valid_end,
        "test": test_end,
        "job-train": job_train_end,
        "job-light_plan": job_light_end,
        "scale": scale_end,
        "synthetic_plan": synthetic_end
    }

    for phase in ['synthetic_plan', 'job-light_plan']:
        evaluate_xgb(xgb, 0, ends[phase], directory, phase)
        print('-'*100)
