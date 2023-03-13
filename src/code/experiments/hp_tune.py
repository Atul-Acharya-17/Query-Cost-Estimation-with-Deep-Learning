# Only for lgbm
# Decide values to tune.

import argparse

hp = {
    'boosting_type': ['gbdt', 'dart'],
    'num_leaves': [x for x in range(10, 101, 1)],
    'max_depth': [x for x in range(6, 33, 2)],
    'n_estimators': [x for x in range(50, 301, 1)],
    'colsample_bytree': [x / 100 for x in range(60, 101, 5)],
    'min_child_samples': [x for x in range(5, 30, 5)],
    'reg_alpha': [x / 100 for x in range(0, 21, 1)],
    'reg_lambda': [x / 100 for x in range(0, 21, 1)],
    'learning_rate': [0.1] + [x / 100 for x in range(5, 101, 5)]
}

import argparse
import pandas as pd
import numpy as np
import time
import json
import math
import torch

import xgboost
import lightgbm
import warnings

import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

import pickle

from ..plan.utils import unnormalize, unnormalize_log
from ..plan.entities import PredicateNodeVector, PlanNodeVector
from ..train.loss_fn import q_error
from ..plan.utils import obtain_upper_bound_query_size, obtain_upper_bound_query_size_log
from ..train.helpers import get_batch_job_tree
from ..plan.map import physic_ops_id, compare_ops_id, bool_ops_id, id2op

from ..networks.tree_lstm import TreeLSTMBatch
from ..networks.tree_gbm import TreeGBM

from ..constants import DATA_ROOT, RESULT_ROOT, NUM_TRAIN, NUM_VAL, NUM_TEST, BATCH_SIZE, JOB_LIGHT, JOB_TRAIN, SCALE, SYNTHETIC

print(xgboost.__version__)
print(lightgbm.__version__)

from ..plan.utils import class2json


def q_error_loss(pred, target, mini, maxi):

    # Only for imdb dataset
    pred = unnormalize_log(pred, mini=mini, maxi=maxi)
    target = unnormalize_log(target, mini=mini, maxi=maxi)    

    q_err = q_error(pred, target)
    return q_err


def q_loss(pred, target, mini, maxi):
    pred = unnormalize_log(pred, mini=mini, maxi=maxi)
    # print(pred, target)
    q_err = q_error(pred, target)
    return q_err


def flatten_plan(node, tree_pooler):
    op_type = np.array(node.operator_vec)
    feature = np.array(node.extra_info_vec)
    bitmap = np.array(node.sample_vec) * node.has_cond

    cond1 = node.condition1_root
    cond2 = node.condition2_root

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

    cost = np.array(node.cost, dtype="float64")
    card = np.array(node.cardinality, dtype="float64")


    right_card = np.array([1.0])
    right_cost = np.array([0.0])
    left_card = np.array([1.0])
    left_cost = np.array([0.0])

    has_left = np.array([0])
    has_right = np.array([0])

    if len(node.children) == 1: #  Only left child
        x_data, y_cost, y_card, left_cost, left_card = flatten_plan(node.children[0], tree_pooler)
        has_left = np.array([1])

    elif len(node.children) == 2: # 2 children
        x_data_left, y_cost_left, y_card_left, left_cost, left_card = flatten_plan(node.children[0], tree_pooler)
        x_data_right, y_cost_right, y_card_right, right_cost, right_card = flatten_plan(node.children[1], tree_pooler)

        x_data = x_data_left + x_data_right
        y_cost = y_cost_left + y_cost_right
        y_card = y_card_left + y_card_right

        has_left = np.array([1])
        has_right = np.array([1])

    else:
        x_data = []
        y_cost = []
        y_card = []

    data = np.concatenate((op_type, feature, bitmap, condition1_vector, condition2_vector, left_cost, right_cost, left_card, right_card, has_left, has_right))
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


def train_tree_gbm(train_start, train_end, directory, phase, model, method='lgbm', hp_dict={'max_depth':6, 'n_estimators':100, 'eta':0.3, 'learning_rate':0.1, 'num_leaves':31}):

    X_train, y_cost, y_card = generate_training_data(train_start, train_end, directory=directory, phase=phase, tree_pooler=model.pool)

    X_train = np.concatenate(X_train, axis=0)
    y_cost = np.concatenate(y_cost, axis=0).reshape(-1, 1)
    y_card = np.concatenate(y_card, axis=0).reshape(-1, 1)

    n_estimators = hp_dict['n_estimators']
    eta = hp_dict['eta']
    learning_rate = hp_dict['learning_rate']
    max_depth = hp_dict['max_depth']
    num_leaves = hp_dict['num_leaves']
    


    # Generate TreeGBM
    if method == 'xgb':
        gbm_card = xgboost.XGBRegressor(seed=0, n_estimators=n_estimators, max_depth=max_depth, eta=eta, num_leaves=num_leaves)
        gbm_cost = xgboost.XGBRegressor(seed=0, n_estimators=n_estimators, max_depth=max_depth, eta=eta, num_leaves=num_leaves)
    else:
        gbm_card = lightgbm.LGBMRegressor(seed=0, n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, num_leaves=num_leaves)
        gbm_cost = lightgbm.LGBMRegressor(seed=0, n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, num_leaves=num_leaves)

    
    start_time = time.time()
    gbm_cost.fit(X_train, y_cost)
    gbm_card.fit(X_train, y_card)

    end_time = time.time()
    training_time = end_time - start_time

    return gbm_cost, gbm_card, training_time


def evaluate_gbm(gbm, start_idx, end_idx, directory, phase, key):
    cost_losses = []
    card_losses = []
    
    inference_times = []

    cost_preds = []
    cost_actual = []

    card_preds = []
    card_actual = []

    use_true = False
    use_db_pred = False

    for batch_idx in range(start_idx, end_idx + 1):
        input_batch, target_cost, target_cardinality, true_cost, true_card = get_batch_job_tree(batch_idx, phase=phase, directory=directory, get_unnorm=True)

        for idx in range(len(input_batch)):
            plan = input_batch[idx]

            real_cost = true_cost[idx].item()
            real_card = true_card[idx].item()

            start_time = time.time()
            estimated_cost, estimated_card = gbm.predict(plan, use_db_pred=use_db_pred, use_true=use_true)
            end_time = time.time()

            # print(estimated_cost)
            # print(real_cost)

            cost_loss = q_loss(estimated_cost[0], real_cost, cost_label_min, cost_label_max)
            card_loss = q_loss(estimated_card[0], real_card, card_label_min, card_label_max)

            cost_losses.append(cost_loss)
            card_losses.append(card_loss)
                
            inference_times.append(end_time - start_time)

            cost_actual.append(real_cost)
            card_actual.append(real_card)

            cost_preds.append(unnormalize_log(estimated_cost[0], mini=cost_label_min, maxi=cost_label_max))
            card_preds.append(unnormalize_log(estimated_card[0], mini=card_label_min, maxi=card_label_max))

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

    if key not in pred_dict:
        pred_dict[key] = {}

    if phase not in pred_dict[key]:
        pred_dict[key][phase] = {}

    pred_dict[key][phase]['cost_metrics'] = cost_metrics
    pred_dict[key][phase]['card_metrics'] = card_metrics
    pred_dict[key][phase]['time_metrics'] = time_metrics


    #print(f"cost metrics: {cost_metrics}, \ncardinality metrics: {card_metrics}, \nInference Time metrics: {time_metrics}")
    print(f"{round(cost_metrics['mean'], 2)} & {round(cost_metrics['median'], 2)} & {round(cost_metrics['90th'], 2)} & {round(cost_metrics['95th'], 2)} & {round(cost_metrics['99th'], 2)} & {round(cost_metrics['max'], 2)}")
    print(f"{round(card_metrics['mean'], 2)} & {round(card_metrics['median'], 2)} & {round(card_metrics['90th'], 2)} & {round(card_metrics['95th'], 2)} & {round(card_metrics['99th'], 2)} & {round(card_metrics['max'], 2)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', default=100, type=int)
    parser.add_argument('--method', default='xgb')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    pred_dict = {}
    time_dict = {}
    dataset='imdb'
    size=100000
    args = parse_args()

    method=args.method

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

    directory = str(DATA_ROOT) + "/" + dataset + "/workload/tree_data"

    phase=f'train_plan_{100000}'

    train_size = size // BATCH_SIZE - 1 if size % BATCH_SIZE == 0 else size // BATCH_SIZE


    max_depth = hp['max_depth']

    num_models = 5

    train_end = NUM_TRAIN // BATCH_SIZE - 1 if NUM_TRAIN % BATCH_SIZE == 0 else NUM_TRAIN // BATCH_SIZE
    valid_end = NUM_VAL // BATCH_SIZE - 1 if NUM_VAL % BATCH_SIZE == 0 else NUM_VAL // BATCH_SIZE
    test_end = NUM_TEST // BATCH_SIZE - 1 if NUM_TEST % BATCH_SIZE == 0 else NUM_VAL // BATCH_SIZE

    job_train_end = JOB_TRAIN // BATCH_SIZE - 1 if JOB_TRAIN % BATCH_SIZE == 0 else JOB_TRAIN // BATCH_SIZE
    job_light_end = JOB_LIGHT // BATCH_SIZE - 1 if JOB_LIGHT % BATCH_SIZE == 0 else JOB_LIGHT // BATCH_SIZE
    scale_end = SCALE // BATCH_SIZE - 1 if SCALE % BATCH_SIZE == 0 else SCALE // BATCH_SIZE
    synthetic_end = SYNTHETIC // BATCH_SIZE - 1 if SYNTHETIC % BATCH_SIZE == 0 else SYNTHETIC // BATCH_SIZE

    for depth in max_depth:
        print('Hyperparameter: ', depth)
        gbm = TreeGBM(tree_pooler=model.pool, fast_inference=True)
        times = []
        for i in range(num_models):
            hp_dict = { 'max_depth': 6,'n_estimators': 100,'eta': 0.3,'learning_rate': 0.1,'num_leaves': 31 }
            hp_dict['max_depth'] = depth
            gbm_cost, gbm_card, train_time = train_tree_gbm(i*(int(train_size/num_models)), (i+1)*int(train_size/num_models), directory, 'train_plan_100000', model, method, hp_dict=hp_dict)
            print(f'Time to train model {i+1}: {train_time}')
            times.append(train_time)
            gbm.add_estimators(gbm_cost, gbm_card)

        time_dict[depth] = times

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
            print('\n', phase)
            evaluate_gbm(gbm, 0, ends[phase], directory, phase, depth)
        print('-'*100)

    x_data = max_depth
    y_data_synthetic_mean = [cost['mean'] for cost in [pred_dict[d]['synthetic_plan']['cost_metrics'] for d in x_data]]
    y_data_job_light_mean = [cost['mean'] for cost in [pred_dict[d]['job-light_plan']['cost_metrics'] for d in x_data]]


    plt.plot(x_data, y_data_synthetic_mean, label='Synthetic', marker='x')
    plt.plot(x_data, y_data_job_light_mean, label='JOB-ligth', marker='x')
    plt.ylabel('Cost Errors Mean')
    plt.xlabel('Num Training samples')
    plt.legend(['Synthetic500', 'JOB-light'])

    plt.show()

    x_data = max_depth
    y_data_synthetic_mean = [card['mean'] for card in [pred_dict[d]['synthetic_plan']['card_metrics'] for d in x_data]]
    y_data_job_light_mean = [card['mean'] for card in [pred_dict[d]['job-light_plan']['card_metrics'] for d in x_data]]

    plt.plot(x_data, y_data_synthetic_mean, label='Synthetic', marker='x')
    plt.plot(x_data, y_data_job_light_mean, label='JOB-ligth', marker='x')
    plt.ylabel('Cardinality Errors Mean')
    plt.xlabel('Num Training samples')
    plt.legend(['Synthetic500', 'JOB-light'])

    plt.show()
