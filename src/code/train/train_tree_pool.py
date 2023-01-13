import argparse
import pickle

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from ..plan.utils import unnormalize
from ..plan.entities import PredicateNodeVector, PlanNodeVector
from .loss_fn import q_error
from ..plan.map import physic_ops_id, compare_ops_id, bool_ops_id
from ..constants import DATA_ROOT, NUM_TRAIN, NUM_VAL, NUM_TEST, BATCH_SIZE
from ..plan.utils import obtain_upper_bound_query_size
from ..networks.tree_pool import TreePool
from .helpers import get_batch_job_tree


def q_error_loss(pred, target, mini, maxi):
    pred = unnormalize(pred, mini=mini, maxi=maxi)
    target = unnormalize(target, mini=mini, maxi=maxi)
    q_err = q_error(pred, target)
    return q_err


def train(train_start, train_end, validate_start, validate_end, num_epochs, directory):

    hidden_dim = 128
    mlp_hid_dim = 256
    model = TreePool(physic_op_total_num, bool_ops_total_num + compare_ops_total_num + column_total_num + max_string_dim, hidden_dim, mlp_hid_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
        model.train()
        cost_loss_total = 0.
        card_loss_total = 0.
        cards = []
        costs = []
        num_samples = 0
        model.train()
        for batch_idx in range(train_start, train_end + 1):
            input_batch, target_cost, target_cardinality = get_batch_job_tree(batch_idx, phase='train', directory=directory)
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

                cards.append(card_loss.item())
                costs.append(cost_loss.item())

            loss /= len(input_batch)
            loss.backward()
            optimizer.step()

        print("Epoch {}, training cost loss: {}, training card loss: {}".format(epoch, cost_loss_total/num_samples, card_loss_total/num_samples))

        valid_cost_loss, valid_card_loss = validate(model, validate_start, validate_end, directory, phase='valid')
        print("Epoch {}, validation cost loss: {}, validation card loss: {}".format(epoch, valid_cost_loss, valid_card_loss))

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

    avg_cost_loss = sum(cost_losses) / len(cost_losses)
    avg_card_loss = sum(card_losses) / len(card_losses)

    return avg_cost_loss, avg_card_loss


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
        from ..dataset.census13 import columns_id, indexes_id, tables_id, max_string_dim

    elif dataset == 'forest10':
        from ..dataset.forest10 import columns_id, indexes_id, tables_id, max_string_dim

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

    train_end = NUM_TRAIN // BATCH_SIZE - 1 if NUM_TRAIN % BATCH_SIZE == 0 else NUM_TRAIN // BATCH_SIZE
    valid_end = NUM_VAL // BATCH_SIZE - 1 if NUM_VAL % BATCH_SIZE == 0 else NUM_VAL // BATCH_SIZE
    test_end = NUM_TEST // BATCH_SIZE - 1 if NUM_TEST % BATCH_SIZE == 0 else NUM_VAL // BATCH_SIZE
    
    epochs = 200

    model = train(0, train_end, 0, valid_end, epochs, directory=directory)
    cost_loss, card_loss = validate(model, 0, test_end, directory, 'test')
    print(f"test cost loss: {cost_loss}, test cardinality loss: {card_loss}")