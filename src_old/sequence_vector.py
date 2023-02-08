from ..src.code.plan.utils import encode_sample
from ..src.code.plan.map import physic_ops_id, compare_ops_id, bool_ops_id
import argparse
import numpy as np
import re
import torch
import math
import json
import itertools
import os

from ..src.code.constants import DATA_ROOT


def encode_condition_op(condition_op, relation_name, index_name):
    # bool_operator + left_value + compare_operator + right_value
    if condition_op == None:
        vec = [0 for _ in range(condition_op_dim)]
    elif condition_op['op_type'] == 'Bool':
        idx = bool_ops_id[condition_op['operator']]
        vec = [0 for _ in range(bool_ops_total_num)]
        vec[idx-1] = 1
    else:
        operator = condition_op['operator']
        left_value = condition_op['left_value']
        if re.match(r'.+\..+', left_value) == None:
            if relation_name == None:
                relation_name = index_name.split(left_value)[1].strip('_')
            left_value = relation_name + '.' + left_value
        else:
            relation_name = left_value.split('.')[0]
        left_value_idx = columns_id[left_value]
        left_value_vec = [0 for _ in range(column_total_num)]
        left_value_vec[left_value_idx-1] = 1
        right_value = condition_op['right_value']
        column_name = left_value.split('.')[1]
        if re.match(r'^[a-z][a-zA-Z0-9_]*\.[a-z][a-zA-Z0-9_]*$', right_value) != None and right_value.split('.')[0] in data:
            operator_idx = compare_ops_id[operator]
            operator_vec = [0 for _ in range(compare_ops_total_num)]
            operator_vec[operator_idx-1] = 1
            right_value_idx = columns_id[right_value]
            right_value_vec = [0]
            left_value_vec[right_value_idx-1] = 1
        elif data[relation_name].dtypes[column_name] == 'int64' or data[relation_name].dtypes[column_name] == 'float64':
            right_value = float(right_value)
            value_max = min_max_column[relation_name][column_name]['max']
            value_min = min_max_column[relation_name][column_name]['min']
            right_value_vec = [(right_value - value_min) / (value_max - value_min)]
            operator_idx = compare_ops_id[operator]
            operator_vec = [0 for _ in range(compare_ops_total_num)]
            operator_vec[operator_idx-1] = 1            
        elif re.match(r'^__LIKE__', right_value) != None:
            operator_idx = compare_ops_id['~~']
            operator_vec = [0 for _ in range(compare_ops_total_num)]
            operator_vec[operator_idx-1] = 1
            right_value = right_value.strip('\'')[8:]
            right_value_vec = get_representation(right_value, left_value).tolist()
        elif re.match(r'^__NOTLIKE__', right_value) != None:
            operator_idx = compare_ops_id['!~~']
            operator_vec = [0 for _ in range(compare_ops_total_num)]
            operator_vec[operator_idx-1] = 1
            right_value = right_value.strip('\'')[11:]
            right_value_vec = get_representation(right_value, left_value).tolist()
        elif re.match(r'^__NOTEQUAL__', right_value) != None:
            operator_idx = compare_ops_id['!=']
            operator_vec = [0 for _ in range(compare_ops_total_num)]
            operator_vec[operator_idx-1] = 1
            right_value = right_value.strip('\'')[12:]
            right_value_vec = get_representation(right_value, left_value).tolist()
        elif re.match(r'^__ANY__', right_value) != None:
            operator_idx = compare_ops_id['=']
            operator_vec = [0 for _ in range(compare_ops_total_num)]
            operator_vec[operator_idx-1] = 1
            right_value = right_value.strip('\'')[7:].strip('{}')
            right_value_vec = []
            count = 0
            for v in right_value.split(','):
                v = v.strip('"').strip('\'')
                if len(v) > 0:
                    count += 1
                    vec = get_representation(v, left_value).tolist()
                    if len(right_value_vec) == 0:
                        right_value_vec = [0 for _ in vec]
                    for idx, vv in enumerate(vec):
                        right_value_vec[idx] += vv
            for idx in range(len(right_value_vec)):
                right_value_vec[idx] /= len(right_value.split(','))
        elif right_value == 'None':
            operator_idx = compare_ops_id['!Null']
            operator_vec = [0 for _ in range(compare_ops_total_num)]
            operator_vec[operator_idx-1] = 1
            if operator == 'IS':
                right_value_vec = [1]
            elif operator == '!=':
                right_value_vec = [0]
            else:
                print (operator)
                raise

        else:
            operator_idx = compare_ops_id[operator]
            operator_vec = [0 for _ in range(compare_ops_total_num)]
            operator_vec[operator_idx-1] = 1
            left_value = left_value.split('.')[1]
            right_value_vec = list(get_representation(right_value, left_value))
        vec = [0 for _ in range(bool_ops_total_num)]
        vec = vec + left_value_vec + operator_vec + right_value_vec
    num_pad = condition_op_dim - len(vec)
    result = np.pad(vec, (0, num_pad), 'constant')
    return result

def bitand(sample1, sample2):
    return np.minimum(sample1, sample2)

def encode_node_job(node, condition_max_num):
    # operator + first_condition + second_condition + relation
    extra_info_num = max(column_total_num, table_total_num, index_total_num)
    operator_vec = np.array([0 for _ in range(physic_op_total_num)])
    
    extra_info_vec = np.array([0 for _ in range(extra_info_num)])
    condition1_vec = np.array([[0 for _ in range(condition_op_dim)] for _ in range(condition_max_num)])
    condition2_vec = np.array([[0 for _ in range(condition_op_dim)] for _ in range(condition_max_num)])
    ### Samples Starts
    sample_vec = np.array([1 for _ in range(1000)])
    ### Samples Ends
    has_condition = 0
    if node != None:
        operator = node['node_type']
        operator_idx = physic_ops_id[operator]
        operator_vec[operator_idx-1] = 1
        if operator == 'Materialize' or operator == 'BitmapAnd' or operator == 'Result':
            pass
        elif operator == 'Sort':
            for key in node['sort_keys']:
                extra_info_inx = columns_id[key]
                extra_info_vec[extra_info_inx-1] = 1
        elif operator == 'Hash Join' or operator == 'Merge Join' or operator == 'Nested Loop':
            condition1_vec = encode_condition(node['condition'], None, None, condition_max_num)
        elif operator == 'Aggregate':
            for key in node['group_keys']:
                extra_info_inx = columns_id[key]
                extra_info_vec[extra_info_inx-1] = 1
        elif operator == 'Seq Scan' or operator == 'Bitmap Heap Scan' or operator == 'Index Scan' or operator == 'Bitmap Index Scan' or operator == 'Index Only Scan':
            relation_name = node['relation_name']
            index_name = node['index_name']
            if relation_name != None:
                extra_info_inx = tables_id[relation_name]
            else:
                extra_info_inx = indexes_id[index_name]
            extra_info_vec[extra_info_inx-1] = 1
            condition1_vec = encode_condition(node['condition_filter'], relation_name, index_name, condition_max_num)
            condition2_vec = encode_condition(node['condition_index'], relation_name, index_name, condition_max_num)
            if 'bitmap' in node:
                ### Samples Starts
                sample_vec = encode_sample(node['bitmap'])
                ### Samples Ends
                has_condition = 1
            if 'bitmap_filter' in node:
                ### Samples Starts
                sample_vec = bitand(encode_sample(node['bitmap_filter']), sample_vec)
                ### Samples Ends
                has_condition = 1
            if 'bitmap_index' in node:
                ### Samples Starts
                sample_vec = bitand(encode_sample(node['bitmap_index']), sample_vec)
                ### Samples Ends
                has_condition = 1

    cardinality = node['cardinality'] if 'cardinality' in node else 0
    cost = node['cost'] if 'cost' in node else 0

    return operator_vec, extra_info_vec, condition1_vec, condition2_vec, sample_vec, has_condition, cost, cardinality

def encode_condition(condition, relation_name, index_name, condition_max_num):
    # print(condition) 
    # exit()
    if len(condition) == 0:
        vecs = [[0 for _ in range(condition_op_dim)]]
    else:
        vecs = [encode_condition_op(condition_op, relation_name, index_name) for condition_op in condition]
    num_pad = condition_max_num - len(vecs)
    result = np.pad(vecs, ((0, num_pad),(0,0)), 'constant')
    return result


class TreeNode(object):
    def __init__(self, current_vec, parent, idx, level_id):
        self.item = current_vec
        self.idx = idx
        self.level_id = level_id
        self.parent = parent
        self.children = []
    def get_parent(self):
        return self.parent
    def get_item(self):
        return self.item
    def get_children(self):
        return self.children
    def add_child(self, child):
        self.children.append(child)
    def get_idx(self):
        return self.idx
    def __str__(self):
        return 'level_id: ' + self.level_id + '; idx: ' + self.idx

def recover_tree(vecs, parent, start_idx):
    if len(vecs) == 0:
        return vecs, start_idx
    if vecs[0] == None:
        return vecs[1:], start_idx+1
    node = TreeNode(current_vec=vecs[0], parent=parent, idx=start_idx, level_id=-1)
    while True:
        vecs, start_idx = recover_tree(vecs[1:], node, start_idx+1)
        parent.add_child(node)
        if len(vecs) == 0:
            return vecs, start_idx
        if vecs[0] == None:
            return vecs[1:], start_idx+1
        node = TreeNode(current_vec=vecs[0], parent=parent, idx=start_idx, level_id=-1)

def dfs_tree_to_level(root, level_id, nodes_by_level):
    root.level_id = level_id
    if len(nodes_by_level) <= level_id:
        nodes_by_level.append([])
    nodes_by_level[level_id].append(root)
    root.idx = len(nodes_by_level[level_id])
    for c in root.get_children():
        dfs_tree_to_level(c, level_id+1, nodes_by_level)

def encode_plan_job(plan, condition_max_num=100):
    operators, extra_infos, condition1s, condition2s, samples, condition_masks, costs, cardinalities = [], [], [], [], [], [], [], []
    mapping = []
    
    nodes_by_level = []
    node = TreeNode(current_vec=plan[0], parent=None, idx=0, level_id=-1)
    recover_tree(plan[1:], node, 1)
    dfs_tree_to_level(node, 0, nodes_by_level)
    
    for level in nodes_by_level:
        operators.append([])
        extra_infos.append([])
        condition1s.append([])
        condition2s.append([])
        samples.append([])
        condition_masks.append([])
        mapping.append([])
        costs.append([])
        cardinalities.append([])

        for node in level:
            operator, extra_info, condition1, condition2, sample, condition_mask, cost, cardinality = encode_node_job(node.item, condition_max_num)
            operators[-1].append(operator)
            extra_infos[-1].append(extra_info)
            condition1s[-1].append(condition1)
            condition2s[-1].append(condition2)
            samples[-1].append(sample)
            condition_masks[-1].append(condition_mask)
            costs[-1].append(cost)
            cardinalities[-1].append(cardinality)
            if len(node.children) == 2:
                mapping[-1].append([n.idx for n in node.children])
            elif len(node.children) == 1:
                mapping[-1].append([node.children[0].idx, 0])
            else:
                mapping[-1].append([0, 0])

    return operators, extra_infos, condition1s, condition2s, samples, condition_masks, mapping, costs, cardinalities


def normalize_label(labels, mini, maxi):
    labels_norm = (labels - mini) / (maxi - mini)
    labels_norm = torch.min(labels_norm, torch.ones_like(labels_norm))
    labels_norm = torch.max(labels_norm, torch.zeros_like(labels_norm))
    return labels_norm

def unnormalize(vecs, mini, maxi):
    return (vecs * (maxi - mini) + mini)

def obtain_upper_bound_query_size(path):
    plan_node_max_num = 0
    condition_max_num = 0
    cost_label_max = 0.0
    cost_label_min = 9999999999.0
    card_label_max = 0.0
    card_label_min = 9999999999.0
    plans = []
    with open(path, 'r') as f:
        for plan in f.readlines():
            plan = json.loads(plan)
            plans.append(plan)
            cost = [plan['cost']]
            cardinality = [plan['cardinality']]

            for seq in plan['seq']:
                if seq == None:
                    continue
                if 'cardinality' in seq:
                    cardinality.append(seq['cardinality'])
                if 'cost' in seq:
                    cost.append(seq['cost'])

            if max(cost) > cost_label_max:
                cost_label_max = max(cost)
            elif min(cost) < cost_label_min:
                cost_label_min = min(cost)
            if max(cardinality) > card_label_max:
                card_label_max = max(cardinality)
            elif min(cardinality) < card_label_min:
                card_label_min = min(cardinality)
            sequence = plan['seq']
            plan_node_num = len(sequence)
            if plan_node_num > plan_node_max_num:
                plan_node_max_num = plan_node_num
            for node in sequence:
                if node == None:
                    continue
                if 'condition_filter' in node:
                    condition_num = len(node['condition_filter'])
                    if condition_num > condition_max_num:
                        condition_max_num = condition_num
                if 'condition_index' in node:
                    condition_num = len(node['condition_index'])
                    if condition_num > condition_max_num:
                        condition_max_num = condition_num
    # cost_label_min, cost_label_max = math.log(cost_label_min), math.log(cost_label_max)
    # card_label_min, card_label_max = math.log(card_label_min), math.log(card_label_max)
    print (plan_node_max_num, condition_max_num)
    print (cost_label_min, cost_label_max)
    print (card_label_min, card_label_max)
    return plan_node_max_num, condition_max_num, cost_label_min, cost_label_max, card_label_min, card_label_max

def merge_plans_level(level1, level2, isMapping=False):
    for idx, level in enumerate(level2):
        if idx >= len(level1):
            level1.append([])
        if isMapping:
            if idx < len(level1) - 1:
                base = len(level1[idx+1])
                for i in range(len(level)):
                    if level[i][0] > 0:
                        level[i][0] += base
                    if level[i][1] > 0:
                        level[i][1] += base
        level1[idx] += level
    return level1


def make_data_job(plans):
    target_cost_batch = []
    target_card_batch = []
    operators_batch = []
    extra_infos_batch = []
    condition1s_batch = []
    condition2s_batch = []
    node_masks_batch = []
    samples_batch = []
    condition_masks_batch = []
    mapping_batch = []
    intermediate_cost_batch = []
    intermediate_card_batch = []
    
    for plan in plans:
        target_cost = plan['cost']
        target_cardinality = plan['cardinality']
        target_cost_batch.append(target_cost)
        target_card_batch.append(target_cardinality)
        plan = plan['seq']
        operators, extra_infos, condition1s, condition2s, samples, condition_masks, mapping, cost, cardinality = encode_plan_job(plan, condition_max_num)
        
        operators_batch = merge_plans_level(operators_batch, operators)
        extra_infos_batch = merge_plans_level(extra_infos_batch, extra_infos)
        condition1s_batch = merge_plans_level(condition1s_batch, condition1s)
        condition2s_batch = merge_plans_level(condition2s_batch, condition2s)
        samples_batch = merge_plans_level(samples_batch, samples)
        condition_masks_batch = merge_plans_level(condition_masks_batch, condition_masks)
        mapping_batch = merge_plans_level(mapping_batch, mapping, True)
        intermediate_cost_batch = merge_plans_level(intermediate_cost_batch, cost)
        intermediate_card_batch = merge_plans_level(intermediate_card_batch, cardinality)

    max_nodes = 0
    for o in operators_batch:
        if len(o) > max_nodes:
            max_nodes = len(o)

    operators_batch = np.array([np.pad(v, ((0, max_nodes - len(v)),(0,0)), 'constant') for v in operators_batch])
    extra_infos_batch = np.array([np.pad(v, ((0, max_nodes - len(v)),(0,0)), 'constant') for v in extra_infos_batch])
    condition1s_batch = np.array([np.pad(v, ((0, max_nodes - len(v)),(0,0),(0,0)), 'constant') for v in condition1s_batch])
    condition2s_batch = np.array([np.pad(v, ((0, max_nodes - len(v)),(0,0),(0,0)), 'constant') for v in condition2s_batch])
    samples_batch = np.array([np.pad(v, ((0, max_nodes - len(v)),(0,0)), 'constant') for v in samples_batch])
    condition_masks_batch = np.array([np.pad(v, (0, max_nodes - len(v)), 'constant') for v in condition_masks_batch])
    mapping_batch = np.array([np.pad(v, ((0, max_nodes - len(v)),(0,0)), 'constant') for v in mapping_batch])

    intermediate_cost_batch = np.array([(np.pad(v, (0, max_nodes - len(v)),'constant')) for v in intermediate_cost_batch])
    intermediate_card_batch = np.array([(np.pad(v, (0, max_nodes - len(v)),'constant'))for v in intermediate_card_batch])

    print ('operators_batch: ', operators_batch.shape)
    
    target_cost_batch = torch.FloatTensor(target_cost_batch)
    target_card_batch = torch.FloatTensor(target_card_batch)
    operators_batch = torch.FloatTensor([operators_batch])
    extra_infos_batch = torch.FloatTensor([extra_infos_batch])
    condition1s_batch = torch.FloatTensor([condition1s_batch])
    condition2s_batch = torch.FloatTensor([condition2s_batch])
    samples_batch = torch.FloatTensor([samples_batch])
    condition_masks_batch = torch.FloatTensor([condition_masks_batch])
    mapping_batch = torch.FloatTensor([mapping_batch])
    intermediate_cost_batch = torch.FloatTensor(intermediate_cost_batch)
    intermediate_card_batch = torch.FloatTensor(intermediate_card_batch)
    
    target_cost_batch = normalize_label(target_cost_batch, cost_label_min, cost_label_max)
    target_card_batch = normalize_label(target_card_batch, card_label_min, card_label_max)
    intermediate_cost_batch = normalize_label(intermediate_cost_batch, cost_label_min, cost_label_max)
    intermediate_card_batch = normalize_label(intermediate_card_batch, card_label_min, card_label_max)
    
    return target_cost_batch, target_card_batch, operators_batch, extra_infos_batch, condition1s_batch, condition2s_batch, samples_batch, condition_masks_batch, mapping_batch, intermediate_card_batch, intermediate_cost_batch

def chunks(arr, batch_size):
    return [arr[i:i+batch_size] for i in range(0, len(arr), batch_size)]


def save_data_job(plans, batch_size=64, phase='train', dataset='census13'):
    suffix = phase + "_"
    batch_id = 0
    directory=f'{DATA_ROOT}/{dataset}/workload/seq_data'

    if not os.path.exists(directory):
        os.makedirs(directory)

    for batch_id, plans_batch in enumerate(chunks(plans, batch_size)):
        print ('batch_id', batch_id, len(plans_batch))
        target_cost_batch, target_cardinality_batch, operators_batch, extra_infos_batch, condition1s_batch, condition2s_batch, samples_batch, condition_masks_batch, mapping_batch, intermediate_card, intermediate_cost = make_data_job(plans_batch)
        np.save(directory+'/target_cost_'+suffix+str(batch_id)+'.np', target_cost_batch.numpy())
        np.save(directory+'/target_cardinality_'+suffix+str(batch_id)+'.np', target_cardinality_batch.numpy())
        np.save(directory+'/operators_'+suffix+str(batch_id)+'.np', operators_batch.numpy())
        np.save(directory+'/extra_infos_'+suffix+str(batch_id)+'.np', extra_infos_batch.numpy())
        np.save(directory+'/condition1s_'+suffix+str(batch_id)+'.np', condition1s_batch.numpy())
        np.save(directory+'/condition2s_'+suffix+str(batch_id)+'.np', condition2s_batch.numpy())
        np.save(directory+'/samples_'+suffix+str(batch_id)+'.np', samples_batch.numpy())
        np.save(directory+'/condition_masks_'+suffix+str(batch_id)+'.np', condition_masks_batch.numpy())
        np.save(directory+'/mapping_'+suffix+str(batch_id)+'.np', mapping_batch.numpy())
        np.save(directory+'/intermediate_card_'+suffix+str(batch_id)+'.np', intermediate_card.numpy())
        np.save(directory+'/intermediate_cost_'+suffix+str(batch_id)+'.np', intermediate_cost.numpy())
        print ('saved: ', str(batch_id))

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
        from ..src.code.dataset.census13 import columns_id, indexes_id, tables_id, get_representation, data, min_max_column


    index_total_num = len(indexes_id)
    table_total_num = len(tables_id)
    column_total_num = len(columns_id)
    physic_op_total_num = len(physic_ops_id)
    compare_ops_total_num = len(compare_ops_id)
    bool_ops_total_num = len(bool_ops_id)
    condition_op_dim = bool_ops_total_num + compare_ops_total_num + column_total_num + 1000
    condition_op_dim_pro = bool_ops_total_num + column_total_num + 3

    plan_node_max_num, condition_max_num, cost_label_min, cost_label_max, card_label_min, card_label_max = obtain_upper_bound_query_size(str(DATA_ROOT) + "/" + dataset + "/workload/plans/" + "train_plans_encoded.json")

    plans = []
    with open(str(DATA_ROOT) + "/" + dataset + "/workload/plans/" + "train_plans_encoded.json") as f:
        for idx, seq in enumerate(f.readlines()):
            plan = json.loads(seq)
            plans.append(plan)

    save_data_job(plans=plans, batch_size=64, phase='train', dataset='census13')

