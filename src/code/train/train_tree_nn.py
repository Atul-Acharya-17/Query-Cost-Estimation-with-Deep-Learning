import argparse
import pandas as pd
import numpy as np
import time
import json
import math

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from ..plan.utils import unnormalize, unnormalize_log
from ..plan.entities import PredicateNodeVector, PlanNodeVector
from .loss_fn import q_error
from ..plan.map import physic_ops_id, compare_ops_id, bool_ops_id
from ..constants import DATA_ROOT, RESULT_ROOT, NUM_TRAIN, NUM_VAL, NUM_TEST, BATCH_SIZE, JOB_LIGHT, JOB_TRAIN, SCALE, SYNTHETIC
from ..plan.utils import obtain_upper_bound_query_size, obtain_upper_bound_query_size_log
from ..networks.tree_lstm import TreeLSTM, TreeLSTMBatch
from ..networks.tree_nn import TreeNN, TreeNNBatch
from ..networks.tree_gru import TreeGRU, TreeGRUBatch
from ..networks.attn import TreeAttnBatch
from ..networks.tree_nn_skip import TreeSkip
from .helpers import get_batch_job_tree

torch.autograd.set_detect_anomaly(True)

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

def train_batch(train_start, train_end, validate_start, validate_end, num_epochs, directory, phase='train', val=False, val_phase='valid', method='tree_lstm'):

    start = time.time()
    hidden_dim = 128
    mlp_hid_dim = 256

    if method == 'tree_nn':
        print('Using TreeNN')
        model = TreeNNBatch(physic_op_total_num, bool_ops_total_num + compare_ops_total_num + column_total_num + max_string_dim, feature_num, hidden_dim, mlp_hid_dim, embedding_type=embedding_type)
        best_model = TreeNNBatch(physic_op_total_num, bool_ops_total_num + compare_ops_total_num + column_total_num + max_string_dim, feature_num, hidden_dim, mlp_hid_dim, embedding_type=embedding_type)

    elif method == 'tree_gru':
        print('Using TreeGRU')
        model = TreeGRUBatch(physic_op_total_num, bool_ops_total_num + compare_ops_total_num + column_total_num + max_string_dim, feature_num, hidden_dim, mlp_hid_dim, embedding_type=embedding_type)
        best_model = TreeGRUBatch(physic_op_total_num, bool_ops_total_num + compare_ops_total_num + column_total_num + max_string_dim, feature_num, hidden_dim, mlp_hid_dim, embedding_type=embedding_type)
    
    elif method == 'tree_lstm':
        print('Using TreeLSTM')
        model = TreeLSTMBatch(physic_op_total_num, bool_ops_total_num + compare_ops_total_num + column_total_num + max_string_dim, feature_num, hidden_dim, mlp_hid_dim, embedding_type=embedding_type)
        best_model = TreeLSTMBatch(physic_op_total_num, bool_ops_total_num + compare_ops_total_num + column_total_num + max_string_dim, feature_num, hidden_dim, mlp_hid_dim, embedding_type=embedding_type)

    elif method == 'tree_attn':
        print('Using TreeAttn')
        model = TreeAttnBatch(physic_op_total_num, bool_ops_total_num + compare_ops_total_num + column_total_num + max_string_dim, feature_num, hidden_dim, mlp_hid_dim, embedding_type=embedding_type)
        best_model = TreeAttnBatch(physic_op_total_num, bool_ops_total_num + compare_ops_total_num + column_total_num + max_string_dim, feature_num, hidden_dim, mlp_hid_dim, embedding_type=embedding_type)

    elif method == 'tree_skip':
        print('Using TreeSkip')
        model = TreeSkip(physic_op_total_num, bool_ops_total_num + compare_ops_total_num + column_total_num + max_string_dim, feature_num, hidden_dim, mlp_hid_dim, embedding_type=embedding_type)
        best_model = TreeSkip(physic_op_total_num, bool_ops_total_num + compare_ops_total_num + column_total_num + max_string_dim, feature_num, hidden_dim, mlp_hid_dim, embedding_type=embedding_type)

    else:
        raise NotImplementedError

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    cost_loss_train = []
    cost_loss_val = []

    card_loss_train = []
    card_loss_val = []
    
    best_state = model.state_dict()

    best_loss = math.inf

    for epoch in range(num_epochs):
        model.train()
        cost_loss_total = 0.
        card_loss_total = 0.

        num_samples = 0

        for batch_idx in range(train_start, train_end + 1):
            input_batch, target_cost, target_cardinality = get_batch_job_tree(batch_idx, phase=phase, directory=directory)
            target_cost, target_cardinality= torch.FloatTensor(target_cost), torch.FloatTensor(target_cardinality)
            target_cost, target_cardinality = Variable(target_cost), Variable(target_cardinality)

            optimizer.zero_grad()

            num_samples += len(input_batch)

            plans = input_batch

            estimated_cost, estimated_card = model(plans)

            cost_loss = sum([q_error_loss(estimated_cost[idx], target_cost[idx], cost_label_min, cost_label_max) for idx in range(len(input_batch))])
            card_loss = sum([q_error_loss(estimated_card[idx], target_cardinality[idx], card_label_min, card_label_max) for idx in range(len(input_batch))])

            loss = cost_loss + card_loss

            cost_loss_total += cost_loss.item()
            card_loss_total += card_loss.item()

            loss /= len(input_batch)
            loss.backward()
            optimizer.step()

        print("Epoch {}, training cost loss: {}, training card loss: {}".format(epoch, cost_loss_total/num_samples, card_loss_total/num_samples))

        cost_loss_train.append(cost_loss_total / num_samples)
        card_loss_train.append(card_loss_total / num_samples)

        if val:
            valid_cost_losses, valid_card_losses, _ = validate(model, validate_start, validate_end, directory, phase=val_phase, entire_batch=True)
            avg_cost_loss = sum(valid_cost_losses) / len(valid_cost_losses)
            avg_card_loss = sum(valid_card_losses) / len(valid_card_losses)

            print("Epoch {}, validation cost loss: {}, validation card loss: {}".format(epoch, avg_cost_loss, avg_card_loss))

            cost_loss_val.append(avg_cost_loss)
            card_loss_val.append(avg_card_loss)

            if avg_cost_loss + avg_card_loss < best_loss:
                print(f"Saving state of epoch: {epoch}")
                best_state = model.state_dict()


    end = time.time()
    print(f"Total: {end - start}")

    best_model.load_state_dict(best_state)    

    return model, best_model, cost_loss_train, cost_loss_val, card_loss_train, card_loss_val

def train(train_start, train_end, validate_start, validate_end, num_epochs, directory, phase='train', val=True, val_phase='valid'):

    start = time.time()
    hidden_dim = 128
    mlp_hid_dim = 128
    model = TreeLSTM(physic_op_total_num, bool_ops_total_num + compare_ops_total_num + column_total_num + max_string_dim, feature_num, hidden_dim, mlp_hid_dim, embedding_type=embedding_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    cost_loss_train = []
    cost_loss_val = []

    card_loss_train = []
    card_loss_val = []

    for epoch in range(num_epochs):
        model.train()
        cost_loss_total = 0.
        card_loss_total = 0.
        # cards = []
        # costs = []
        num_samples = 0
        for batch_idx in range(train_start, train_end + 1):
            input_batch, target_cost, target_cardinality = get_batch_job_tree(batch_idx, phase=phase, directory=directory)
            target_cost, target_cardinality= torch.FloatTensor(target_cost), torch.FloatTensor(target_cardinality)
            target_cost, target_cardinality = Variable(target_cost), Variable(target_cardinality)

            optimizer.zero_grad()

            num_samples += len(input_batch)

            loss = 0.0

            for idx in range(len(input_batch)):
                plan = input_batch[idx]
                cost = target_cost[idx]
                card = target_cardinality[idx]

                estimate_cost, estimate_cardinality = model(plan)

                cost_loss = q_error_loss(estimate_cost[0], cost, cost_label_min, cost_label_max)
                card_loss = q_error_loss(estimate_cardinality[0], card, card_label_min, card_label_max)

                loss += cost_loss + card_loss
                cost_loss_total += cost_loss.item()
                card_loss_total += card_loss.item()

            loss /= len(input_batch)
            loss.backward()
            optimizer.step()

        print("Epoch {}, training cost loss: {}, training card loss: {}".format(epoch, cost_loss_total/num_samples, card_loss_total/num_samples))

        if val:
            valid_cost_losses, valid_card_losses, _ = validate(model, validate_start, validate_end, directory, phase=val_phase)
            avg_cost_loss = sum(valid_cost_losses) / len(valid_cost_losses)
            avg_card_loss = sum(valid_card_losses) / len(valid_card_losses)

            print("Epoch {}, validation cost loss: {}, validation card loss: {}".format(epoch, avg_cost_loss, avg_card_loss))

    end = time.time()

    print(f"Total: {end - start}")
    return model, cost_loss_train, cost_loss_val, card_loss_train, card_loss_val

def validate(model, start_idx, end_idx, directory, phase='valid', batch=True, entire_batch=False):
    model.eval()

    cost_losses = []
    card_losses = []
    
    inference_times = []

    num_samples=0

    for batch_idx in range(start_idx, end_idx + 1):
        input_batch, target_cost, target_cardinality, true_cost, true_card = get_batch_job_tree(batch_idx, phase=phase, directory=directory, get_unnorm=True)
        target_cost, target_cardinality= torch.FloatTensor(target_cost), torch.FloatTensor(target_cardinality)
        target_cost, target_cardinality = Variable(target_cost), Variable(target_cardinality)


        if not entire_batch:
            for idx in range(len(input_batch)):
                plan = input_batch[idx]
                cost = true_cost[idx]
                card = true_card[idx]

                if batch:
                    plan = [plan]
                
                start_time = time.time()
                estimate_cost, estimate_cardinality = model(plan, batch=True)
                end_time = time.time()
                # target_cost = true_cost[idx]
                # target_cardinality = true_card[idx]

                cost_loss = q_loss(estimate_cost[0], cost, cost_label_min, cost_label_max)
                card_loss = q_loss(estimate_cardinality[0], card, card_label_min, card_label_max)
                
                cost_losses.append(cost_loss.item())
                card_losses.append(card_loss.item())
                
                inference_times.append(end_time - start_time)

        else:
            num_samples += len(input_batch)

            plans = input_batch

            estimated_cost, estimated_card = model(plans)

            cost_loss = [q_loss(estimated_cost[idx], target_cost[idx], cost_label_min, cost_label_max).item() for idx in range(len(input_batch))]
            card_loss = [q_loss(estimated_card[idx], target_cardinality[idx], card_label_min, card_label_max).item() for idx in range(len(input_batch))]

            cost_losses += cost_loss
            card_losses += card_loss

    return cost_losses, card_losses, inference_times


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='random')
    parser.add_argument('--name', default='tree_lstm')
    parser.add_argument('--embedding-type', default='tree_pool')
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--size', default=10000, type=int)
    parser.add_argument('--method', default='tree_lstm')

    args = parser.parse_args()
    return args


def val_and_print(model, test_end, directory, phase, name='tree_lstm'):
    cost_losses, card_losses, inference_times = validate(model, 0, test_end, directory, phase)

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
    stats_df.to_csv(str(RESULT_ROOT) + "/output/" + dataset + f"/results_{name}_{phase}.csv")

def save_losses(cost_loss_train, cost_loss_val, card_loss_train, card_loss_val, dataset, name, phase):
    json_data = {
        'cost_loss_train':cost_loss_train,
        'cost_loss_val':cost_loss_val,
        'card_loss_train':card_loss_train,
        'card_loss_val':card_loss_val
    }
    
    file_path = str(RESULT_ROOT) + "/output/" + dataset + f"/losses_{name}_{phase}.json"

    with open(file_path, 'w') as f:
        json.dump(json_data, f)

if __name__ == '__main__':
    args = parse_args()
    dataset = args.dataset
    name = args.name

    method=args.method

    epochs = args.epochs

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

    elif dataset == 'imdb':
        from ..dataset.imdb import columns_id, indexes_id, tables_id, max_string_dim


    phases = ['test']
    train_path = "train_plans"
    

    if dataset == "imdb":
        phases = ['synthetic_plan', 'job-light_plan']
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

    directory = str(DATA_ROOT) + "/" + dataset + "/workload/tree_data/"

    train_end = NUM_TRAIN // BATCH_SIZE - 1 if NUM_TRAIN % BATCH_SIZE == 0 else NUM_TRAIN // BATCH_SIZE
    valid_end = NUM_VAL // BATCH_SIZE - 1 if NUM_VAL % BATCH_SIZE == 0 else NUM_VAL // BATCH_SIZE
    test_end = NUM_TEST // BATCH_SIZE - 1 if NUM_TEST % BATCH_SIZE == 0 else NUM_VAL // BATCH_SIZE

    job_train_end = JOB_TRAIN // BATCH_SIZE - 1 if JOB_TRAIN % BATCH_SIZE == 0 else JOB_TRAIN // BATCH_SIZE
    job_light_end = JOB_LIGHT // BATCH_SIZE - 1 if JOB_LIGHT % BATCH_SIZE == 0 else JOB_LIGHT // BATCH_SIZE
    scale_end = SCALE // BATCH_SIZE - 1 if SCALE % BATCH_SIZE == 0 else SCALE // BATCH_SIZE
    synthetic_end = SYNTHETIC // BATCH_SIZE - 1 if SYNTHETIC % BATCH_SIZE == 0 else SYNTHETIC // BATCH_SIZE

    ends = {
        "train": train_end,
        "valid": valid_end,
        "test": test_end,
        "job-train": job_train_end,
        "job-light_plan": job_light_end,
        "scale": scale_end,
        "synthetic_plan": synthetic_end
    }
    

    if dataset != "imdb":

        model, cost_loss_train, cost_loss_val, card_loss_train, card_loss_val = train_batch(0, train_end, 0, valid_end, epochs, directory=directory, val=True)

        for phase in phases:
            val_and_print(model, ends[phase], directory, phase, name)
            print('-'*25)

    else:
        size = args.size
        train_size = size // BATCH_SIZE - 1 if size % BATCH_SIZE == 0 else size // BATCH_SIZE
        train_end = int(0.9 * train_size)
        valid_start = train_end
        valid_end = train_size
        model, best_model, cost_loss_train, cost_loss_val, card_loss_train, card_loss_val = train_batch(0, train_end, valid_start, valid_end, epochs, directory=directory, phase=f'train_plan_{size}', val=True, val_phase=f'train_plan_{size}', method=method)

        save_losses(cost_loss_train, cost_loss_val, card_loss_train, card_loss_val, dataset, name, phase='train')

        for phase in phases:
            val_and_print(model, ends[phase], directory, phase, name=name)
            print('-'*100)


        for phase in phases:
            val_and_print(best_model, ends[phase], directory, phase, name='best_' + name)
            print('-'*100)
        

        # torch.save(model.state_dict(), str(RESULT_ROOT) + "/models/" + dataset + f"/{name}.pt")