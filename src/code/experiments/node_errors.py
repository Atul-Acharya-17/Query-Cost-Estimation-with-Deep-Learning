import argparse
import pandas as pd
import numpy as np
import time
import json
import math
import torch

import matplotlib.pyplot as plt

import xgboost
import lightgbm
import warnings


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


def q_error_loss(pred, target, mini, maxi):

    if dataset == 'imdb':
        pred = unnormalize_log(pred, mini=mini, maxi=maxi)
        target = unnormalize_log(target, mini=mini, maxi=maxi)    
    else:    
        pred = unnormalize(pred, mini=mini, maxi=maxi)
        target = unnormalize(target, mini=mini, maxi=maxi)

    q_err = q_error(pred, target)
    return q_err


def q_loss(pred, target, mini, maxi):
    pred = unnormalize_log(pred, mini=mini, maxi=maxi)
    q_err = q_error(pred, target)
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

    has_left = np.array([0])
    has_right = np.array([0])

    if len(plan.children) == 1: #  Only left child
        x_data, y_cost, y_card, left_cost, left_card = flatten_plan(plan.children[0], tree_pooler)
        has_left = np.array([1])

    elif len(plan.children) == 2: # 2 children
        x_data_left, y_cost_left, y_card_left, left_cost, left_card = flatten_plan(plan.children[0], tree_pooler)
        x_data_right, y_cost_right, y_card_right, right_cost, right_card = flatten_plan(plan.children[1], tree_pooler)

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



def generate_data(train_start, train_end, directory, phase, tree_pooler):
    X = []
    y_cost = []
    y_card = []
    for batch_idx in range(train_start, train_end + 1):
        print(f'{batch_idx + 1}/{train_end}', end='\r')
        input_batch, target_cost, target_cardinality = get_batch_job_tree(batch_idx, phase=phase, directory=directory)
        for plan in input_batch:
            x, cost, card, _, _ = flatten_plan(plan, tree_pooler)
            X += x
            y_cost += cost
            y_card += card

    return X, y_cost, y_card



def train_gbm(train_start, train_end, directory, phase, model):
    X_train, y_cost, y_card = generate_data(train_start, train_end, directory=directory, phase=phase, tree_pooler=model.pool)

    X_train = np.concatenate(X_train, axis=0)
    y_cost = np.concatenate(y_cost, axis=0).reshape(-1, 1)
    y_card = np.concatenate(y_card, axis=0).reshape(-1, 1)

    print(X_train.shape)

    if method == 'xgb':
        gbm_card = xgboost.XGBRegressor(n_estimators=100, max_depth=16, eta=0.1, subsample=0.7, colsample_bytree=0.8, seed=0)
        gbm_cost = xgboost.XGBRegressor(n_estimators=100, max_depth=16, eta=0.1, subsample=0.7, colsample_bytree=0.8, seed=0)

    else:
        gbm_card = lightgbm.LGBMRegressor(n_estimators=200, max_depth=16, learning_rate=0.6, subsample=0.7, colsample_bytree=0.8, seed=0)
        gbm_cost = lightgbm.LGBMRegressor(n_estimators=200, max_depth=16, learning_rate=0.6, subsample=0.7, colsample_bytree=0.8, seed=0)

    start_time = time.time()
    gbm_cost.fit(X_train, y_cost)
    gbm_card.fit(X_train, y_card)

    end_time = time.time()
    training_time = end_time - start_time

    return gbm_cost, gbm_card, training_time



def evaluate_gmb_nodes(cost_models, card_models, pooler, start_idx, end_idx, directory, phase):

    X, y_cost, y_card = generate_data(start_idx, end_idx, directory, phase, pooler)

    X = np.concatenate(X, axis=0)

    cost_preds = np.zeros(X.shape[0])
    card_preds = np.zeros(X.shape[0])

    for model in cost_models:
        cost_preds += model.predict(X)

    for model in card_models:
        card_preds += model.predict(X)

    cost_preds /= len(cost_models)
    card_preds /= len(card_models)

    q_errors_cost = {}
    q_errors_card = {}

    for idx in range(cost_preds.shape[0]):
    
        operation_vec = X[idx][0:physic_op_total_num]

        node_type = id2op[np.argmax(operation_vec) + 1]

        cost_loss = q_error_loss(cost_preds[idx], y_cost[idx], cost_label_min, cost_label_max)
        card_loss = q_error_loss(card_preds[idx], y_card[idx], card_label_min, card_label_max)

        if node_type not in q_errors_cost:
            q_errors_cost[node_type] = []

        if node_type not in q_errors_card:
            q_errors_card[node_type] = []

        q_errors_cost[node_type].append(cost_loss)

        q_errors_card[node_type].append(card_loss)

    return q_errors_cost, q_errors_card



def print_stats(errors):
    for key, item in errors.items():
        max = float(np.max(item))
        median = float(np.median(item))
        mean = float(np.mean(item))

        count = len(item)

        print(f"{key} & {count} & {mean} & {median} & {max} \\\\")

def print_table(cost_errors, card_errors):
    cost_errors = dict(sorted(cost_errors.items()))
    card_errors = dict(sorted(card_errors.items()))

    for key in cost_errors.keys():
        median_cost = float(np.median(cost_errors[key]))
        mean_cost = float(np.mean(cost_errors[key]))

        median_card = float(np.median(card_errors[key]))
        mean_card = float(np.mean(card_errors[key]))

        count = len(cost_errors[key])

        print(f"{key} & {count} & {round(mean_cost, 2)} & {round(median_cost, 2)} & {round(mean_card, 2)} & {round(median_card, 2)} \\\\")


def plot_boxplot(errors:dict):

    error_list = [[float(x) for x in values] for values in errors.values()]

    plt.boxplot(error_list, labels=list(errors.keys()), vert=False, showfliers=False)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='imdb')
    parser.add_argument('--test-set', default='job-light_plans')
    parser.add_argument('--method', default='lgbm')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    
    dataset = args.dataset
    method = args.method

    job_train_end = JOB_TRAIN // BATCH_SIZE - 1 if JOB_TRAIN % BATCH_SIZE == 0 else JOB_TRAIN // BATCH_SIZE
    job_light_end = JOB_LIGHT // BATCH_SIZE - 1 if JOB_LIGHT % BATCH_SIZE == 0 else JOB_LIGHT // BATCH_SIZE
    scale_end = SCALE // BATCH_SIZE - 1 if SCALE % BATCH_SIZE == 0 else SCALE // BATCH_SIZE
    synthetic_end = SYNTHETIC // BATCH_SIZE - 1 if SYNTHETIC % BATCH_SIZE == 0 else SYNTHETIC // BATCH_SIZE

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

    phase=f'train_plan_{100000}'

    gbm = TreeGBM(tree_pooler=model.pool, fast_inference=False)

    size = 100000
    train_size = size // BATCH_SIZE - 1 if size % BATCH_SIZE == 0 else size // BATCH_SIZE

    times = []
    
    model_dir = str(RESULT_ROOT) + '/models/' + dataset + '/' + method

    num_models = 5
    for i in range(num_models):
        gbm_cost, gbm_card, train_time = train_gbm(i*(int(train_size/num_models)), (i+1)*int(train_size/num_models), directory, phase, model)
        print(f'Time to train model {i+1}: {train_time}')
        times.append(train_time)

        gbm.add_estimators(gbm_cost, gbm_card)

    ends = {
        "job-light_plan": job_light_end,
        "synthetic_plan": synthetic_end
    }

    for phase in ["job-light_plan", "synthetic_plan"]:

        print()
        print(phase)
    
        cost_errors, card_errors = evaluate_gmb_nodes(gbm.cost_gbm, gbm.card_gbm, model.pool, 0, ends[phase], directory, phase)


        print("Cost Errors")
        print_stats(cost_errors)
        print()
        print("Cardinality Errors")
        print_stats(card_errors)

        plot_boxplot(cost_errors)
        plot_boxplot(card_errors)
        print()

        print("Table")
        print_table(cost_errors=cost_errors, card_errors=card_errors)