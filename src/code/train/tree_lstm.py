from ..networks.TreeLSTM import TreeLSTM
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import time
from ..plan.map import physic_ops_id, compare_ops_id, bool_ops_id
from ..constants import DATA_ROOT
from ..plan.encode_plan import obtain_upper_bound_query_size
import numpy as np

def normalize_label(labels, mini, maxi):
    labels_norm = (labels - mini) / (maxi - mini)
    labels_norm = torch.min(labels_norm, torch.ones_like(labels_norm))
    labels_norm = torch.max(labels_norm, torch.zeros_like(labels_norm))
    return labels_norm

def unnormalize(vecs, mini, maxi):
    return (vecs * (maxi - mini) + mini)

def qerror_loss(preds, targets, mini, maxi):
    qerror = []
    preds = unnormalize(preds, mini, maxi)
    targets = unnormalize(targets, mini, maxi)

    try:
        for i in range(len(targets)):
            for j in range(len(targets[i])):
                if preds[i][j] == 0 and targets[i][j] == 0:
                    qerror.append(1)
                elif preds[i][j] == 0:
                    qerror.append(targets[i][j])
                elif targets[i][j] == 0:
                    qerror.append(preds[i][j])
                elif (preds[i][j] > targets[i][j]):
                    qerror.append(preds[i][j]/targets[i][j])
                else:
                    qerror.append(targets[i][j]/preds[i][j])
        return torch.mean(torch.stack(qerror)), torch.median(torch.stack(qerror)), torch.max(torch.stack(qerror)), torch.argmax(torch.stack(qerror))

    except Exception as e:
        for i in range(len(targets)):
            if preds[i] == 0 and targets[i] == 0:
                qerror.append(1)
            elif preds[i] == 0:
                qerror.append(targets[i])
            elif targets[i] == 0:
                qerror.append(preds[i])
            elif (preds[i] > targets[i]):
                qerror.append(preds[i]/targets[i])
            else:
                qerror.append(targets[i]/preds[i])

        # for x in qerror:
        #     print(x)
        # print(sum(qerror))
        return torch.mean(torch.cat(qerror)), torch.median(torch.cat(qerror)), torch.max(torch.cat(qerror)), torch.argmax(torch.cat(qerror))

def get_batch_job(batch_id, phase, directory):
    suffix = phase + '_'
    target_cost_batch = np.load(directory+'/target_cost_'+suffix+str(batch_id)+'.np.npy')
    target_cardinality_batch = np.load(directory+'/target_cardinality_'+suffix+str(batch_id)+'.np.npy')
    operators_batch = np.load(directory+'/operators_'+suffix+str(batch_id)+'.np.npy')
    extra_infos_batch = np.load(directory+'/extra_infos_'+suffix+str(batch_id)+'.np.npy')
    condition1s_batch = np.load(directory+'/condition1s_'+suffix+str(batch_id)+'.np.npy')
    condition2s_batch = np.load(directory+'/condition2s_'+suffix+str(batch_id)+'.np.npy')
    samples_batch = np.load(directory+'/samples_'+suffix+str(batch_id)+'.np.npy')
    condition_masks_batch = np.load(directory+'/condition_masks_'+suffix+str(batch_id)+'.np.npy')
    mapping_batch = np.load(directory+'/mapping_'+suffix+str(batch_id)+'.np.npy')
    intermediate_card = np.load(directory+'/intermediate_card_'+suffix+str(batch_id)+'.np.npy')
    intermediate_cost = np.load(directory+'/intermediate_cost_'+suffix+str(batch_id)+'.np.npy')
    return target_cost_batch, target_cardinality_batch, operators_batch, extra_infos_batch, condition1s_batch, condition2s_batch, samples_batch, condition_masks_batch, mapping_batch, intermediate_card, intermediate_cost

def train(train_start, train_end, validate_start, validate_end, num_epochs, directory):
    input_dim = condition_op_dim
    hidden_dim = 128
    hid_dim = 256
    model = TreeLSTM(input_dim, hidden_dim, hid_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    start = time.time()
    for epoch in range(num_epochs):
        cost_loss_total = 0.
        card_loss_total = 0.
        model.train()
        for batch_idx in range(train_start, train_end):
            target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping, intermediate_card, intermediate_cost = get_batch_job(batch_idx, phase='train', directory=directory)
            target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping, intermediate_card, intermediate_cost = torch.FloatTensor(target_cost), torch.FloatTensor(target_cardinality),torch.FloatTensor(operatorss),torch.FloatTensor(extra_infoss),torch.FloatTensor(condition1ss),torch.FloatTensor(condition2ss), torch.FloatTensor(sampless), torch.FloatTensor(condition_maskss), torch.FloatTensor(mapping), torch.FloatTensor(intermediate_card), torch.FloatTensor(intermediate_cost)
            operatorss, extra_infoss, condition1ss, condition2ss, condition_maskss = operatorss.squeeze(0), extra_infoss.squeeze(0), condition1ss.squeeze(0), condition2ss.squeeze(0), condition_maskss.squeeze(0).unsqueeze(2)
            sampless = sampless.squeeze(0)
            mapping = mapping.squeeze(0)
            target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss = Variable(target_cost), Variable(target_cardinality), Variable(operatorss), Variable(extra_infoss), Variable(condition1ss), Variable(condition2ss)
            sampless = Variable(sampless)
            intermediate_card = Variable(intermediate_card)
            intermediate_cost = Variable(intermediate_cost)
            optimizer.zero_grad()

            estimate_cost, estimate_cardinality, estimate_intermediate_cost, estimate_intermediate_card = model(operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping)
            target_cost = target_cost
            target_cardinality = target_cardinality
            # cost_loss,cost_loss_median,cost_loss_max,cost_max_idx = qerror_loss(estimate_intermediate_cost, intermediate_cost, cost_label_min, cost_label_max)
            # card_loss,card_loss_median,card_loss_max,card_max_idx = qerror_loss(estimate_intermediate_card, intermediate_card, card_label_min, card_label_max)
            # print (card_loss.item(),card_loss_median.item(),card_loss_max.item(),card_max_idx.item())
            # print (cost_loss.item(),cost_loss_median.item(),cost_loss_max.item(),cost_max_idx.item())
            cost_loss,cost_loss_median,cost_loss_max,cost_max_idx = qerror_loss(estimate_cost, target_cost, cost_label_min, cost_label_max)
            card_loss,card_loss_median,card_loss_max,card_max_idx = qerror_loss(estimate_cardinality, target_cardinality, card_label_min, card_label_max)
            loss = cost_loss + card_loss
            loss.backward()
            optimizer.step()
            cost_loss_total += cost_loss.item()
            card_loss_total += card_loss.item()
            start = time.time()
            end = time.time()
        batch_num = train_end - train_start
        print("Epoch {}, training cost loss: {}, training card loss: {}".format(epoch, cost_loss_total/batch_num, card_loss_total/batch_num))

    #     cost_loss_total = 0.
    #     card_loss_total = 0.
    #     for batch_idx in range(validate_start, validate_end):
    #         print ('batch_idx: ', batch_idx)
    #         target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping = get_batch_job(batch_idx)
    #         target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping = torch.FloatTensor(target_cost), torch.FloatTensor(target_cardinality),torch.FloatTensor(operatorss),torch.FloatTensor(extra_infoss),torch.FloatTensor(condition1ss),torch.FloatTensor(condition2ss), torch.FloatTensor(sampless), torch.FloatTensor(condition_maskss), torch.FloatTensor(mapping)
    #         operatorss, extra_infoss, condition1ss, condition2ss, condition_maskss = operatorss.squeeze(0), extra_infoss.squeeze(0), condition1ss.squeeze(0), condition2ss.squeeze(0), condition_maskss.squeeze(0).unsqueeze(2)
    #         sampless = sampless.squeeze(0)
    #         mapping = mapping.squeeze(0)
    #         target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss = Variable(target_cost), Variable(target_cardinality), Variable(operatorss), Variable(extra_infoss), Variable(condition1ss), Variable(condition2ss)
    #         sampless = Variable(sampless)
    #         estimate_cost,estimate_cardinality = model(operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping)
    #         target_cost = target_cost
    #         target_cardinality = target_cardinality
    #         cost_loss,cost_loss_median,cost_loss_max,cost_max_idx = qerror_loss(estimate_cost, target_cost, cost_label_min, cost_label_max)
    #         card_loss,card_loss_median,card_loss_max,card_max_idx = qerror_loss(estimate_cardinality, target_cardinality, card_label_min, card_label_max)
    #         print (card_loss.item(),card_loss_median.item(),card_loss_max.item(),card_max_idx.item())
    #         loss = cost_loss + card_loss
    #         cost_loss_total += cost_loss.item()
    #         card_loss_total += card_loss.item()
    #     batch_num = validate_end - validate_start
    #     print("Epoch {}, validating cost loss: {}, validating card loss: {}".format(epoch, cost_loss_total/batch_num, card_loss_total/batch_num))
    end = time.time()
    print (end-start)

    cost_loss_total = 0.
    card_loss_total = 0.
    model.eval()
    for batch_idx in range(train_start, train_end):
        target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping, intermediate_card, intermediate_cost = get_batch_job(batch_idx, phase='train', directory=directory)
        target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping, intermediate_card, intermediate_cost = torch.FloatTensor(target_cost), torch.FloatTensor(target_cardinality),torch.FloatTensor(operatorss),torch.FloatTensor(extra_infoss),torch.FloatTensor(condition1ss),torch.FloatTensor(condition2ss), torch.FloatTensor(sampless), torch.FloatTensor(condition_maskss), torch.FloatTensor(mapping), torch.FloatTensor(intermediate_card), torch.FloatTensor(intermediate_cost)
        operatorss, extra_infoss, condition1ss, condition2ss, condition_maskss = operatorss.squeeze(0), extra_infoss.squeeze(0), condition1ss.squeeze(0), condition2ss.squeeze(0), condition_maskss.squeeze(0).unsqueeze(2)
        sampless = sampless.squeeze(0)
        mapping = mapping.squeeze(0)
        target_cost, target_cardinality, operatorss, extra_infoss, condition1ss, condition2ss = Variable(target_cost), Variable(target_cardinality), Variable(operatorss), Variable(extra_infoss), Variable(condition1ss), Variable(condition2ss)
        sampless = Variable(sampless)
        intermediate_card = Variable(intermediate_card)
        intermediate_cost = Variable(intermediate_cost)
        optimizer.zero_grad()

        estimate_cost, estimate_cardinality, estimate_intermediate_cost, estimate_intermediate_card = model(operatorss, extra_infoss, condition1ss, condition2ss, sampless, condition_maskss, mapping)
        target_cost = target_cost
        target_cardinality = target_cardinality
        loss = cost_loss + card_loss
        cost_loss,cost_loss_median,cost_loss_max,cost_max_idx = qerror_loss(estimate_cost, target_cost, cost_label_min, cost_label_max)
        card_loss,card_loss_median,card_loss_max,card_max_idx = qerror_loss(estimate_cardinality, target_cardinality, card_label_min, card_label_max)
        print (card_loss.item(),card_loss_median.item(),card_loss_max.item(),card_max_idx.item())
        print (cost_loss.item(),cost_loss_median.item(),cost_loss_max.item(),cost_max_idx.item())
        cost_loss_total += cost_loss.item()
        card_loss_total += card_loss.item()
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
        from ..dataset.census13 import columns_id, indexes_id, tables_id

    plan_node_max_num, condition_max_num, cost_label_min, cost_label_max, card_label_min, card_label_max = obtain_upper_bound_query_size(str(DATA_ROOT) + "/" + dataset + "/workload/plans/" + "train_plans_encoded.json")

    index_total_num = len(indexes_id)
    table_total_num = len(tables_id)
    column_total_num = len(columns_id)
    physic_op_total_num = len(physic_ops_id)
    compare_ops_total_num = len(compare_ops_id)
    bool_ops_total_num = len(bool_ops_id)
    condition_op_dim = bool_ops_total_num + compare_ops_total_num+column_total_num+1000
    condition_op_dim_pro = bool_ops_total_num + column_total_num + 3

    directory = str(DATA_ROOT) + "/" + dataset + "/workload/seq_data/"

    train(0, 15, 0, 1, 200, directory=directory)