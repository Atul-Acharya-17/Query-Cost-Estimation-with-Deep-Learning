from .utils import encode_sample
from .map import physic_ops_id, compare_ops_id, bool_ops_id
import argparse
import numpy as np
import re
import torch
import math
import json
import itertools
import os
from .utils import class2json
import pickle

from ..constants import DATA_ROOT

class PredicateNode():

    def __init__(self, op_type, operator, left_value=None, right_value=None, children=None):

        self.op_type = op_type
        self.operator = operator
        self.left_value = left_value
        self.right_value = right_value
        self.children = children

    def add_child(self, child):
        if self.children is None:
            self.children = [child]
        else:
            self.children.append(child)


class TreeVector():
    
    def __init__(self, op_type, operator, bool_operator_vector, comp_operator_vector, left_vector, right_vector) -> None:
        self.bool_operator_vector = bool_operator_vector
        self.comp_operator_vector = comp_operator_vector
        self.left_vector = left_vector
        self.right_vector = right_vector

        self.op_type = op_type
        self.operator = operator

        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def get_torch_tensor(self):
        return torch.unsqueeze(torch.FloatTensor(self.bool_operator_vector + self.left_vector + self.comp_operator_vector + self.right_vector),0)

def encode_predicate_tree(root, relation_name, index_name):
    
    bool_op_vector = [0 for _ in range(bool_ops_total_num)]
    comp_op_vector = [0 for _ in range(compare_ops_total_num)]
    left_value_vec = [0 for _ in range(column_total_num)]

    right_value_vec = [0 for _ in range(column_total_num)]

    if root.op_type == 'Bool':
        idx = bool_ops_id[root.operator]
        bool_op_vector[idx-1] = 1

    elif root.op_type == 'Compare':
        operator = root.operator
        idx = compare_ops_id[operator]
        comp_op_vector[idx-1] = 1

        left_value = root.left_value

        if re.match(r'.+\..+', left_value) == None:
            if relation_name == None:
                relation_name = index_name.split(left_value)[1].strip('_')
            left_value = relation_name + '.' + left_value
        else:
            relation_name = left_value.split('.')[0]
            
        left_value_idx = columns_id[left_value]
        left_value_vec[left_value_idx-1] = 1
        right_value = root.right_value
        column_name = left_value.split('.')[1]

        if re.match(r'^[a-z][a-zA-Z0-9_]*\.[a-z][a-zA-Z0-9_]*$', right_value) != None and right_value.split('.')[0] in data:
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
            left_value = left_value.split('.')[1]
            right_value_vec = list(get_representation(right_value, left_value))

    num_pad = max_string_dim - len(right_value_vec)

    right_value_vec = [float(x) for x in np.pad(right_value_vec, (0, num_pad), 'constant')]
    node = TreeVector(root.op_type, root.operator, bool_op_vector, comp_op_vector, left_value_vec, right_value_vec)

    if root.children is not None:
        for child in root.children:

            tree_vector = encode_predicate_tree(child, relation_name, index_name)

            node.add_child(tree_vector)

    return node
    

def bitand(sample1, sample2):
    return [float(x) for x in np.minimum(sample1, sample2)]

class PlanNodeVector():

    def __init__(self, operator_vec, extra_info_vec, condition1_root, condition2_root, sample_vec, has_condition, cost, cardinality) -> None:
        self.operator_vec = operator_vec
        self.extra_info_vec = extra_info_vec
        self.condition1_root = condition1_root
        self.condition2_root = condition2_root
        self.sample_vec = sample_vec
        self.has_cond = has_condition
        self.cost = cost
        self.cardinality = cardinality

        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def get_torch_operation_vector(self):
        return torch.unsqueeze(torch.FloatTensor(self.operator_vec), 0)

    def get_torch_extra_info_vector(self):
        return torch.unsqueeze(torch.FloatTensor(self.extra_info_vec), 0)

    def get_torch_sample_bitmap_vector(self):
        return torch.unsqueeze(torch.FloatTensor(self.sample_vec), 0)


def encode_node_job(node):
    # operator + first_condition + second_condition + relation
    extra_info_num = max(column_total_num, table_total_num, index_total_num)
    operator_vec = [0 for _ in range(physic_op_total_num)]
    
    extra_info_vec = [0 for _ in range(extra_info_num)]
    condition1_tree = None
    condition2_tree = None
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
            condition1_tree = encode_condition(node['condition'], None, None, condition_max_num)
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
            condition1_tree = encode_condition(node['condition_filter'], relation_name, index_name)
            condition2_tree = encode_condition(node['condition_index'], relation_name, index_name)
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

    return operator_vec, extra_info_vec, condition1_tree, condition2_tree, sample_vec, has_condition, cost, cardinality


def predseq_2_tree(condition, idx):
    if idx == len(condition):
        return None, idx
    if condition[idx] == None:
        return None, idx + 1
    while idx < len(condition):
        operator = condition[idx]
        if operator['op_type'] == 'Bool':
            node = PredicateNode(operator['op_type'], operator['operator'], children=[])
        elif operator['op_type'] == 'Compare':
            node = PredicateNode(operator['op_type'], operator['operator'], left_value=operator['left_value'], right_value=operator['right_value'], children=None)
        left_child, next_idx = predseq_2_tree(condition, idx + 1)
        if left_child is not None:
            node.add_child(left_child)
        else:
            return node, next_idx
        right_child, next_idx = predseq_2_tree(condition, next_idx)
        if right_child is not None:
            node.add_child(right_child)
        else:
            return node, next_idx
        return node, next_idx


def encode_condition(condition, relation_name, index_name):

    if len(condition) == 0:
        tree = None

    else:
        predicate_tree, _ = predseq_2_tree(condition, 0)
        tree = encode_predicate_tree(predicate_tree, relation_name, index_name)

    return tree

def planseq_2_treevec(seq, idx):
    if idx == len(seq):
        return None, idx
    if seq[idx] is None:
        return None, idx + 1
    while idx < len(seq):
        
        operator, extra_info, condition1, condition2, sample, condition_mask, cost, cardinality = encode_node_job(seq[idx])
        node = PlanNodeVector(operator_vec=operator, extra_info_vec=extra_info, condition1_root=condition1, condition2_root=condition2, sample_vec=sample, has_condition=condition_mask, cost=cost, cardinality=cardinality)

        left_child, next_idx = planseq_2_treevec(seq, idx + 1)
        if left_child is not None:
            node.add_child(left_child)
        else:
            return node, next_idx
        right_child, next_idx = planseq_2_treevec(seq, next_idx)
        if right_child is not None:
            node.add_child(right_child)
        else:
            return node, next_idx

        return node, next_idx
            

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
            cost = plan['cost']
            cardinality = plan['cardinality']

            if cost > cost_label_max:
                cost_label_max = cost
            elif cost < cost_label_min:
                cost_label_min = cost
            if cardinality > card_label_max:
                card_label_max = cardinality
            elif cardinality < card_label_min:
                card_label_min = cardinality
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

    print (plan_node_max_num, condition_max_num)
    print (cost_label_min, cost_label_max)
    print (card_label_min, card_label_max)
    return plan_node_max_num, condition_max_num, cost_label_min, cost_label_max, card_label_min, card_label_max

def encode_plan_job(plan):
    plan_root, _ = planseq_2_treevec(plan, 0)
    return plan_root

def make_data_job(plans):
    target_cost_batch = []
    target_card_batch = []
    input_batch = []
    
    for plan in plans:
        target_cost = plan['cost']
        target_cardinality = plan['cardinality']
        target_cost_batch.append(target_cost)
        target_card_batch.append(target_cardinality)
        plan = plan['seq']
        plan_tree = encode_plan_job(plan)
        input_batch.append(plan_tree)

    target_cost_batch = torch.FloatTensor(target_cost_batch)
    target_card_batch = torch.FloatTensor(target_card_batch)

    target_cost_batch = normalize_label(target_cost_batch, cost_label_min, cost_label_max)
    target_card_batch = normalize_label(target_card_batch, card_label_min, card_label_max)

    return input_batch, target_cost_batch, target_card_batch

def chunks(arr, batch_size):
    return [arr[i:i+batch_size] for i in range(0, len(arr), batch_size)]


def save_data_job(plans, batch_size=64, phase='train', dataset='census13'):
    suffix = phase + "_"
    batch_id = 0
    directory=f'{DATA_ROOT}/{dataset}/workload/tree_data'

    if not os.path.exists(directory):
        os.makedirs(directory)

    for batch_id, plans_batch in enumerate(chunks(plans, batch_size)):
        print ('batch_id', batch_id, len(plans_batch))
        input_batch, target_cost_batch, target_cardinality_batch = make_data_job(plans_batch)
        with open(f'{directory}/input_batch_{suffix+str(batch_id)}.pkl', 'wb') as handle:
            pickle.dump(input_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{directory}/target_cost_{suffix+str(batch_id)}.pkl', 'wb') as handle:
            pickle.dump(target_cost_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{directory}/target_cardinality_{suffix+str(batch_id)}.pkl', 'wb') as handle:
            pickle.dump(target_cardinality_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)
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
        from ..dataset.census13 import columns_id, indexes_id, tables_id, get_representation, data, min_max_column, max_string_dim


    index_total_num = len(indexes_id)
    table_total_num = len(tables_id)
    column_total_num = len(columns_id)
    physic_op_total_num = len(physic_ops_id)
    compare_ops_total_num = len(compare_ops_id)
    bool_ops_total_num = len(bool_ops_id)
    condition_op_dim = bool_ops_total_num + compare_ops_total_num + column_total_num
    condition_op_dim_pro = bool_ops_total_num + column_total_num + 3

    plan_node_max_num, condition_max_num, cost_label_min, cost_label_max, card_label_min, card_label_max = obtain_upper_bound_query_size(str(DATA_ROOT) + "/" + dataset + "/workload/plans/" + "train_plans_encoded.json")

    phases = ['train', 'valid', 'test']

    for phase in phases:
        plans = []
        with open(str(DATA_ROOT) + "/" + dataset + "/workload/plans/" + f"{phase}_plans_encoded.json") as f:
            for idx, seq in enumerate(f.readlines()):
                plan = json.loads(seq)
                plans.append(plan)

        save_data_job(plans=plans, batch_size=64, phase=phase, dataset='census13')

