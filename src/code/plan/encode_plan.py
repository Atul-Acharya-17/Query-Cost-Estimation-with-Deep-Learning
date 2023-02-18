import argparse
import numpy as np
import re
import torch
import json
import os
import pickle

from ..constants import DATA_ROOT, BATCH_SIZE
from .utils import encode_sample, bitand, normalize_label, obtain_upper_bound_query_size, obtain_upper_bound_query_size_intermediate, normalize_label_log, obtain_upper_bound_query_size_log
from .map import physic_ops_id, compare_ops_id, bool_ops_id
from .entities import PredicateNode, PredicateNodeVector, PlanNodeVector



def predicate_seq_2_tree(condition, idx):
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
        left_child, next_idx = predicate_seq_2_tree(condition, idx + 1)
        if left_child is not None:
            node.add_child(left_child)
        else:
            return node, next_idx
        right_child, next_idx = predicate_seq_2_tree(condition, next_idx)
        if right_child is not None:
            node.add_child(right_child)
        else:
            return node, next_idx
        return node, next_idx


def encode_predicate_tree(root, relation_name, index_name):
    
    bool_op_vector = [0 for _ in range(bool_ops_total_num)]
    comp_op_vector = [0 for _ in range(compare_ops_total_num)]
    left_value_vec = [0 for _ in range(column_total_num)]

    right_value_vec = [0 for _ in range(max_string_dim)]

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

        # if dataset=='imdb':
        #     right_value = float(right_value)
        #     value_max = min_max_column[relation_name][column_name]['max']
        #     value_min = min_max_column[relation_name][column_name]['min']
        #     right_value_vec = [(right_value - value_min) / (value_max - value_min)]
        #     operator_idx = compare_ops_id[operator]
        #     operator_vec = [0 for _ in range(compare_ops_total_num)]
        #     operator_vec[operator_idx-1] = 1   

        if re.match(r'^[a-z][a-zA-Z0-9_]*\.[a-z][a-zA-Z0-9_]*$', right_value) != None and right_value.split('.')[0] in tables_id:
            right_value_idx = columns_id[right_value]
            right_value_vec = [0]
            left_value_vec[right_value_idx-1] = 1
        elif dataset=="imdb" or data[relation_name].dtypes[column_name] == 'int64' or data[relation_name].dtypes[column_name] == 'float64':
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

    
    num_pad = max(max_string_dim - len(right_value_vec), 0)

    right_value_vec = [float(x) for x in np.pad(right_value_vec, (0, num_pad), 'constant')]
    node = PredicateNodeVector(root.op_type, root.operator, bool_op_vector, comp_op_vector, left_value_vec, right_value_vec)

    if root.children is not None:
        for child in root.children:

            tree_vector = encode_predicate_tree(child, relation_name, index_name)

            node.add_child(tree_vector)

    return node


def encode_condition_operation(condition_op, relation_name, index_name):
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
        
        if dataset == "imdb":
            right_value = float(right_value)
            value_max = min_max_column[relation_name][column_name]['max']
            value_min = min_max_column[relation_name][column_name]['min']
            right_value_vec = [(right_value - value_min) / (value_max - value_min)]
            operator_idx = compare_ops_id[operator]
            operator_vec = [0 for _ in range(compare_ops_total_num)]
            operator_vec[operator_idx-1] = 1  
            
        elif re.match(r'^[a-z][a-zA-Z0-9_]*\.[a-z][a-zA-Z0-9_]*$', right_value) != None and right_value.split('.')[0] in data:
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
    result = [float(x) for x in np.pad(vec, (0, num_pad), 'constant')]
    return result


def encode_condition(condition, relation_name, index_name, use_tree=True):
    if use_tree:
        if len(condition) == 0:
            return None
        predicate_tree, _ = predicate_seq_2_tree(condition, 0)
        tree = encode_predicate_tree(predicate_tree, relation_name, index_name)
        return tree

    else:
        if len(condition) == 0:
            vecs = [[0 for _ in range(condition_op_dim)]]
        else:
            vecs = [encode_condition_operation(condition_op, relation_name, index_name) for condition_op in condition]
        num_pad = condition_max_num - len(vecs)
        result = np.pad(vecs, ((0, num_pad),(0,0)), 'constant')
        result = [[float(x) for x in array] for array in result]
        return result


def encode_node(node, use_tree):
    # operator + first_condition + second_condition + relation
    # extra_info_num = max(column_total_num, table_total_num, index_total_num)
    extra_info_num = column_total_num + table_total_num + index_total_num + 1 # For num workers

    column_start = 0
    table_start = column_total_num
    index_start = table_total_num

    operator_vec = [0 for _ in range(physic_op_total_num)]
    
    extra_info_vec = [0 for _ in range(extra_info_num)]

    if use_tree:
        condition1 = None
        condition2 = None

    else:
        condition1 = [[0 for _ in range(condition_op_dim)] for _ in range(condition_max_num)]
        condition2 = [[0 for _ in range(condition_op_dim)] for _ in range(condition_max_num)]
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
                extra_info_vec[column_start + extra_info_inx-1] = 1
        elif operator == 'Hash Join' or operator == 'Merge Join' or operator == 'Nested Loop':
            condition1 = encode_condition(node['condition'], None, None, use_tree=use_tree)
        elif operator == 'Aggregate':
            for key in node['group_keys']:
                extra_info_inx = columns_id[key]
                extra_info_vec[column_start + extra_info_inx-1] = 1
        elif operator == 'Seq Scan' or operator == 'Bitmap Heap Scan' or operator == 'Index Scan' or operator == 'Bitmap Index Scan' or operator == 'Index Only Scan':
            relation_name = node['relation_name']
            index_name = node['index_name']
            if relation_name != None:
                extra_info_inx = table_start + tables_id[relation_name]
            else:
                extra_info_inx = index_start + indexes_id[index_name]
            extra_info_vec[extra_info_inx-1] = 1
            condition1 = encode_condition(node['condition_filter'], relation_name, index_name, use_tree=use_tree)
            condition2 = encode_condition(node['condition_index'], relation_name, index_name, use_tree=use_tree)
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
        elif operator == 'Gather' or operator == 'Gather Merge':
            num_workers = node['workers_planned']
            extra_info_vec[-1] = num_workers


    cardinality = node['cardinality'] if 'cardinality' in node else 0
    cost = node['cost'] if 'cost' in node else 0
    
    db_estimate_card = node['db_estimate_card'] if 'db_estimate_card' in node else 0

    return operator_vec, extra_info_vec, condition1, condition2, sample_vec, has_condition, cost, cardinality, db_estimate_card


def plan_seq_2_tree_vec(seq, idx=0, use_tree=True):
    if idx >= len(seq):
        return None, idx + 1
    if seq[idx] is None:
        return None, idx + 1
        
    operator, extra_info, condition1, condition2, sample, condition_mask, unnorm_cost, unnorm_cardinality, unnorm_db_estimate_card = encode_node(seq[idx], use_tree=use_tree)
    if dataset == 'imdb':
        cardinality = normalize_label_log(torch.FloatTensor([unnorm_cardinality]), card_label_min, card_label_max)
        cost = normalize_label_log(torch.FloatTensor([unnorm_cost]), cost_label_min, cost_label_max)
        db_estimate_card = normalize_label_log(torch.FloatTensor([unnorm_db_estimate_card]), card_label_min, card_label_max)

    else:
        cost = normalize_label(torch.FloatTensor([cost]), cost_label_min, cost_label_max)
        cardinality = normalize_label(torch.FloatTensor([cardinality]), card_label_min, card_label_max)
        db_estimate_card = normalize_label(torch.FloatTensor([db_estimate_card]), card_label_min, card_label_max)
        
    node = PlanNodeVector(operator_vec=operator, extra_info_vec=extra_info, condition1_root=condition1, condition2_root=condition2, sample_vec=sample, has_condition=condition_mask, cost=cost, cardinality=cardinality, db_estimate_card=db_estimate_card, unnorm_card=torch.FloatTensor([unnorm_cardinality]), unnorm_cost=torch.FloatTensor([unnorm_cost]), unnorm_db_estimate_card=torch.FloatTensor([unnorm_db_estimate_card]))

    left_child, next_idx = plan_seq_2_tree_vec(seq, idx + 1, use_tree=use_tree)
    if left_child is not None:
        node.add_child(left_child)
    else:
        return node, next_idx
    right_child, next_idx = plan_seq_2_tree_vec(seq, next_idx, use_tree=use_tree)

    while next_idx < len(seq) and seq[next_idx] is None:
        next_idx += 1
    if right_child is not None:
        node.add_child(right_child)
    else:
        return node, next_idx

    return node, next_idx


def encode_plan(plan, use_tree=True):
    plan_root, _ = plan_seq_2_tree_vec(plan, 0, use_tree=use_tree)
    return plan_root

def make_data_job(plans, use_tree=True):
    target_cost_batch = []
    target_card_batch = []
    input_batch = []

    true_cost_batch = []
    true_card_batch = []
    
    for plan in plans:
        target_cost = plan['cost']
        target_cardinality = plan['cardinality']
        target_cost_batch.append(target_cost)
        target_card_batch.append(target_cardinality)
        true_cost_batch.append(target_cost)
        true_card_batch.append(target_cardinality)

        plan = plan['seq']
        plan_tree = encode_plan(plan, use_tree)
        input_batch.append(plan_tree)

    target_cost_batch = torch.FloatTensor(target_cost_batch)
    target_card_batch = torch.FloatTensor(target_card_batch)

    true_cost_batch = torch.FloatTensor(true_cost_batch)
    true_card_batch = torch.FloatTensor(true_card_batch)

    if dataset == 'imdb':
        target_cost_batch = normalize_label_log(target_cost_batch, cost_label_min, cost_label_max)
        target_card_batch = normalize_label_log(target_card_batch, card_label_min, card_label_max)

    else:   
        target_cost_batch = normalize_label(target_cost_batch, cost_label_min, cost_label_max)
        target_card_batch = normalize_label(target_card_batch, card_label_min, card_label_max)

    return input_batch, target_cost_batch, target_card_batch, true_cost_batch, true_card_batch


def chunks(arr, batch_size):
    return [arr[i:i+batch_size] for i in range(0, len(arr), batch_size)]


def save_data_job(plans, batch_size=BATCH_SIZE, phase='train', dataset='census13', use_tree=True):
    suffix = phase + "_"
    batch_id = 0
    directory=f'{DATA_ROOT}/{dataset}/workload/tree_data'

    if not os.path.exists(directory):
        os.makedirs(directory)

    for batch_id, plans_batch in enumerate(chunks(plans, batch_size)):
        print ('batch_id', batch_id, len(plans_batch))
        input_batch, target_cost_batch, target_cardinality_batch, true_cost_batch, true_card_batch = make_data_job(plans_batch, use_tree)
        with open(f'{directory}/input_batch_{suffix+str(batch_id)}.pkl', 'wb') as handle:
            pickle.dump(input_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{directory}/target_cost_{suffix+str(batch_id)}.pkl', 'wb') as handle:
            pickle.dump(target_cost_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{directory}/target_cardinality_{suffix+str(batch_id)}.pkl', 'wb') as handle:
            pickle.dump(target_cardinality_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{directory}/true_cardinality_{suffix+str(batch_id)}.pkl', 'wb') as handle:
            pickle.dump(true_card_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{directory}/true_cost_{suffix+str(batch_id)}.pkl', 'wb') as handle:
            pickle.dump(true_cost_batch, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        print ('saved: ', str(batch_id))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='random')
    parser.add_argument('--version', default='original')
    parser.add_argument('--name', default='base')
    parser.add_argument('--tree', action='store_true')
    parser.add_argument('--no-tree', dest='tree', action='store_false')
    parser.add_argument('--intermediate', action='store_true')
    parser.add_argument('--no-intermediate', dest='intermediate')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    dataset = args.dataset
    version = args.version
    name = args.name

    use_tree = args.tree
    process_intermediate = args.intermediate

    if dataset == 'census13':
        from ..dataset.census13 import columns_id, indexes_id, tables_id, get_representation, data, min_max_column, max_string_dim

    elif dataset == 'forest10':
        from ..dataset.forest10 import columns_id, indexes_id, tables_id, get_representation, data, min_max_column, max_string_dim

    elif dataset == 'power7':
        from ..dataset.power7 import columns_id, indexes_id, tables_id, get_representation, data, min_max_column, max_string_dim

    elif dataset == 'dmv11':
        from ..dataset.dmv11 import columns_id, indexes_id, tables_id, get_representation, data, min_max_column, max_string_dim

    elif dataset == 'imdb':
        from ..dataset.imdb import columns_id, indexes_id, tables_id, get_representation, min_max_column, max_string_dim


    index_total_num = len(indexes_id)
    table_total_num = len(tables_id)
    column_total_num = len(columns_id)
    physic_op_total_num = len(physic_ops_id)
    compare_ops_total_num = len(compare_ops_id)
    bool_ops_total_num = len(bool_ops_id)
    condition_op_dim = bool_ops_total_num + compare_ops_total_num + column_total_num + max_string_dim
    condition_op_dim_pro = bool_ops_total_num + column_total_num + 3

    train_path = "train_plans"

    if dataset == "imdb":
        train_path = "train_plan_100000"

    if process_intermediate:
        plan_node_max_num, condition_max_num, cost_label_min, cost_label_max, card_label_min, card_label_max = obtain_upper_bound_query_size_intermediate(str(DATA_ROOT) + "/" + dataset + "/workload/plans/" + f"{train_path}_encoded.json")

    else:
        if dataset == 'imdb':
            plan_node_max_num, condition_max_num, cost_label_min, cost_label_max, card_label_min, card_label_max = obtain_upper_bound_query_size_log(str(DATA_ROOT) + "/" + dataset + "/workload/plans/" + f"{train_path}_encoded.json")
        else:
            plan_node_max_num, condition_max_num, cost_label_min, cost_label_max, card_label_min, card_label_max = obtain_upper_bound_query_size(str(DATA_ROOT) + "/" + dataset + "/workload/plans/" + f"{train_path}_encoded.json")

    phases = ['train_plans', 'valid_plans', 'test_plans']

    if dataset == 'imdb':
        phases = ['job-light_plan', 'synthetic_plan', 'train_plan_100000']#'train_plan_500', 'train_plan_1000', 'train_plan_2000', 'train_plan_5000', 'train_plan_10000', 'train_plan_20000', 'train_plan_50000', 'train_plan_100000']

    for phase in phases:
        plans = []
        with open(str(DATA_ROOT) + "/" + dataset + "/workload/plans/" + f"{phase}_encoded.json") as f:
            for idx, seq in enumerate(f.readlines()):
                plan = json.loads(seq)
                plans.append(plan)

        save_data_job(plans=plans, batch_size=BATCH_SIZE, phase=phase, dataset=dataset, use_tree=use_tree)

