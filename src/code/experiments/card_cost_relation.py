import argparse
import pickle
from ..constants import RESULT_ROOT, DATA_ROOT, JOB_LIGHT, BATCH_SIZE, SYNTHETIC_500

from ..plan.utils import unnormalize, unnormalize_log
from ..plan.entities import PredicateNodeVector, PlanNodeVector
from ..train.loss_fn import q_error
from ..plan.utils import obtain_upper_bound_query_size, obtain_upper_bound_query_size_log
from ..train.helpers import get_batch_job_tree
from ..plan.map import physic_ops_id, compare_ops_id, bool_ops_id

from ..networks.tree_lstm import TreeLSTMBatch
from ..networks.tree_gbm import TreeGBM

from ..dataset.imdb import columns_id, indexes_id, tables_id, max_string_dim

import pandas as pd
import numpy as np
import torch
import time


import warnings


warnings.filterwarnings("ignore")


def q_loss(pred, target, mini, maxi):
    pred = unnormalize_log(pred, mini=mini, maxi=maxi)
    q_err = q_error(pred, target)
    return q_err


def evaluate_gbm(gbm, start_idx, end_idx, directory, phase, mode):
    cost_losses = []

    use_true = False
    use_db_pred = False

    if mode == 'use_true':
        use_true=True

    elif mode == 'use_db_pred':
        use_db_pred=True

    for batch_idx in range(start_idx, end_idx + 1):
        input_batch, target_cost, target_cardinality, true_cost, true_card = get_batch_job_tree(batch_idx, phase=phase, directory=directory, get_unnorm=True)

        for idx in range(len(input_batch)):
            plan = input_batch[idx]

            real_cost = true_cost[idx].item()
            real_card = true_card[idx].item()

            estimated_cost, _ = gbm.predict(plan, use_true=use_true, use_db_pred=use_db_pred)

            cost_loss = q_loss(estimated_cost[0], real_cost, cost_label_min, cost_label_max)

            cost_losses.append(cost_loss)

    cost_metrics = {
        'max': np.max(cost_losses),
        '99th': np.percentile(cost_losses, 99),
        '95th': np.percentile(cost_losses, 95),
        '90th': np.percentile(cost_losses, 90),
        'median': np.median(cost_losses),
        'mean': np.mean(cost_losses),
    }

    print(phase, mode, len(cost_losses))



    print(f"cost metrics: {cost_metrics}")
    print(f"{round(cost_metrics['mean'], 2)} & {round(cost_metrics['median'], 2)} & {round(cost_metrics['90th'], 2)} & {round(cost_metrics['95th'], 2)} & {round(cost_metrics['99th'], 2)} & {round(cost_metrics['max'], 2)}")
    # stats_df = pd.DataFrame(list(cost_losses), columns=['cost_errors', 'card_errors', 'inference_time'])
    # stats_df.to_csv(str(RESULT_ROOT) + "/output/" + 'imdb' + f"/results_{name}_{mode}_{phase}.csv")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='lgbm')

    args = parser.parse_args()
    return args



if __name__ == '__main__':

    args = parse_args()

    name = args.name

    cost_models = []
    card_models = []

    MODEL_DIR = str(RESULT_ROOT) + '/models/imdb/' + name

    for i in range(5):

        with open(MODEL_DIR+f'_{i+1}_cost.pickle', "rb") as f:
            print(MODEL_DIR+f'_{i+1}_cost.pickle')
            cost_gbm = pickle.load(f)
        with open(MODEL_DIR+f'_{i+1}_card.pickle', "rb") as f:
            card_gbm = pickle.load(f)

        cost_models.append(cost_gbm)
        card_models.append(card_gbm)

    plan_node_max_num, condition_max_num, cost_label_min, cost_label_max, card_label_min, card_label_max = obtain_upper_bound_query_size_log(str(DATA_ROOT) + "/" + 'imdb' + "/workload/plans/" + f"{'train_plan_100000'}_encoded.json")

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
    pool_path = str(RESULT_ROOT) + '/models/' + 'imdb' + '/tree_lstm_100000.pt'
 
    model.load_state_dict(torch.load(pool_path))

    model.eval()

    gbm = TreeGBM(tree_pooler=model.pool, fast_inference=True)

    print(len(cost_models))
    for idx in range(5):
        gbm.add_estimators(cost_models[i], card_models[i])

    job_light_end = JOB_LIGHT // BATCH_SIZE - 1 if JOB_LIGHT % BATCH_SIZE == 0 else JOB_LIGHT // BATCH_SIZE
    synthetic_end = SYNTHETIC_500 // BATCH_SIZE - 1 if SYNTHETIC_500 % BATCH_SIZE == 0 else SYNTHETIC_500 // BATCH_SIZE

    ends = {
        "job-light_plan": job_light_end,
        "synthetic_plan": synthetic_end
    }

    print(job_light_end)
    print(synthetic_end)

    directory = str(DATA_ROOT) + "/" + "imdb" + "/workload/tree_data/"

    for phase in ['synthetic_plan', 'job-light_plan']:
        
        print(f'{phase}, use_true')
        evaluate_gbm(gbm, 0, ends[phase], directory, phase=phase, mode='use_true')
        print('-'*100)
        
        print(f'{phase}, use_db_pred')
        evaluate_gbm(gbm, 0, ends[phase], directory, phase=phase, mode='use_db_pred')
        print('-'*100)

        print(f'{phase}, use_estimator')
        evaluate_gbm(gbm, 0, ends[phase], directory, phase=phase, mode='use_estimator')
        print('-'*100)


