import argparse
import pickle
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from ..plan.utils import unnormalize
from ..plan.entities import PredicateNodeVector, PlanNodeVector
from .loss_fn import q_error
from ..plan.map import physic_ops_id, compare_ops_id, bool_ops_id
from ..constants import DATA_ROOT, NUM_TRAIN, NUM_VAL, NUM_TEST, BATCH_SIZE, JOB_LIGHT, JOB_TRAIN, SCALE, SYNTHETIC
from ..plan.utils import obtain_upper_bound_query_size
from ..networks.tree_lstm import TreeLSTM
from ..networks.tree_nn import TreeNN
from ..networks.tree_gru import TreeGRU
from .helpers import get_batch_job_tree


def q_error_loss(pred, target, mini, maxi):
    pred = unnormalize(pred, mini=mini, maxi=maxi)
    target = unnormalize(target, mini=mini, maxi=maxi)
    q_err = q_error(pred, target)
    return q_err


def train(train_start, train_end, validate_start, validate_end, num_epochs, directory, phase='train', val=True):

    hidden_dim = 128
    mlp_hid_dim = 256
    model = TreeNN(physic_op_total_num, bool_ops_total_num + compare_ops_total_num + column_total_num + max_string_dim, feature_num, hidden_dim, mlp_hid_dim, embedding_type=embedding_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        cost_loss_total = 0.
        card_loss_total = 0.
        # cards = []
        # costs = []
        num_samples = 0
        model.train()
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

                # cards.append(card_loss.item())
                # costs.append(cost_loss.item())

            loss /= len(input_batch)
            loss.backward()
            optimizer.step()

        print("Epoch {}, training cost loss: {}, training card loss: {}".format(epoch, cost_loss_total/num_samples, card_loss_total/num_samples))

        if val:
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

            estimate_cost, estimate_cardinality = model(plan)
            target_cost = target_cost
            target_cardinality = target_cardinality

            cost_loss = q_error_loss(estimate_cost[0], cost, cost_label_min, cost_label_max)
            card_loss = q_error_loss(estimate_cardinality[0], card, card_label_min, card_label_max)
            
            cost_losses.append(cost_loss.item())
            card_losses.append(card_loss.item())

    return cost_losses, card_losses


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='random')
    parser.add_argument('--version', default='original')
    parser.add_argument('--name', default='base')
    parser.add_argument('--embedding-type', default='tree_pool')
    args = parser.parse_args()
    return args


def val_and_print(model, test_end, directory, phase):
    cost_losses, card_losses = validate(model, 0, test_end, directory, phase)

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

if __name__ == '__main__':
    args = parse_args()
    dataset = args.dataset
    version = args.version
    name = args.name

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

    phases = ['test']
    train_path = "train"

    if dataset == "imdb":
        phases = ['synthetic', 'scale','job-light']
        train_path = "job-train"

    plan_node_max_num, condition_max_num, cost_label_min, cost_label_max, card_label_min, card_label_max = obtain_upper_bound_query_size(str(DATA_ROOT) + "/" + dataset + "/workload/plans/" + f"{train_path}_plans_encoded.json")

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

    job_train_end = JOB_TRAIN // BATCH_SIZE - 1 if JOB_TRAIN % BATCH_SIZE == 0 else JOB_TRAIN // BATCH_SIZE
    job_light_end = JOB_LIGHT // BATCH_SIZE - 1 if JOB_LIGHT % BATCH_SIZE == 0 else JOB_LIGHT // BATCH_SIZE
    scale_end = SCALE // BATCH_SIZE - 1 if SCALE % BATCH_SIZE == 0 else SCALE // BATCH_SIZE
    synthetic_end = SYNTHETIC // BATCH_SIZE - 1 if SYNTHETIC % BATCH_SIZE == 0 else SYNTHETIC // BATCH_SIZE

    ends = {
        "train": train_end,
        "valid": valid_end,
        "test": test_end,
        "job-train": job_train_end,
        "job-light": job_light_end,
        "scale": scale_end,
        "synthetic": synthetic_end
    }
    

    if dataset != "imdb":
        epochs = 50

        model = train(0, train_end, 0, valid_end, epochs, directory=directory)

        for phase in phases:
            val_and_print(model, ends[phase], directory, phase)

    else:
        epochs = 10

        model = train(0, job_train_end, 0, 0, epochs, directory=directory, phase='job-train', val=False)

        for phase in phases:
            val_and_print(model, ends[phase], directory, phase)