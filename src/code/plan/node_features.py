from .node_operations import *
from .utils import change_alias2table
from .predicate_features import pre2seq

import re


def encode_node(node, alias2table):
    relation_name, index_name, cardinality, cost, db_estimate_card = None, None, None, None, None
    if 'Relation Name' in node:
        relation_name = node['Relation Name']
    if 'Index Name' in node:
        index_name = node['Index Name']
    if 'Actual Rows' in node:
        cardinality = node['Actual Rows']
    if 'Actual Total Time' in node:
        cost = node['Actual Total Time']  
    if 'Plan Rows' in node:
        db_estimate_card = node['Plan Rows']
    
    if node['Node Type'] == 'Materialize':
        return Materialize(cost=cost, cardinality=cardinality, db_estimate_card=db_estimate_card), None
    elif node['Node Type'] == 'Hash':
        return Hash(cardinality=cardinality, cost=cost, db_estimate_card=db_estimate_card), None
    elif node['Node Type'] == 'Sort':
        keys = [change_alias2table(key, alias2table) for key in node['Sort Key']]
        return Sort(keys, cost=cost, cardinality=cardinality, db_estimate_card=db_estimate_card), None
    elif node['Node Type'] == 'BitmapAnd':
        return BitmapCombine('BitmapAnd', cardinality=cardinality, cost=cost, db_estimate_card=db_estimate_card), None
    elif node['Node Type'] == 'BitmapOr':
        return BitmapCombine('BitmapOr', cardinality=cardinality, cost=cost, db_estimate_card=db_estimate_card), None
    elif node['Node Type'] == 'Result':
        return Result(cardinality=cardinality, cost=cost, db_estimate_card=db_estimate_card), None
    elif node['Node Type'] == 'Hash Join':
        return Join('Hash Join', pre2seq(node["Hash Cond"], alias2table, relation_name, index_name, True), cardinality=cardinality, cost=cost, db_estimate_card=db_estimate_card), None
    elif node['Node Type'] == 'Merge Join':
        return Join('Merge Join', pre2seq(node["Merge Cond"], alias2table, relation_name, index_name, True), cardinality=cardinality, cost=cost, db_estimate_card=db_estimate_card), None
    elif node['Node Type'] == 'Nested Loop':
        if 'Join Filter' in node:
            condition = pre2seq(node['Join Filter'], alias2table, relation_name, index_name, True)
        else:
            condition = []
        return Join('Nested Loop', condition, cardinality=cardinality, cost=cost, db_estimate_card=db_estimate_card), None
    elif node['Node Type'] == 'Aggregate':
        if 'Group Key' in node:
            keys = [change_alias2table(key, alias2table) for key in node['Group Key']]
        else:
            keys = []
        return Aggregate(node['Strategy'], keys, cardinality=cardinality, cost=cost, db_estimate_card=db_estimate_card), None
    elif node['Node Type'] == 'Seq Scan':
        if 'Filter' in node:
            condition_seq_filter = pre2seq(node['Filter'], alias2table, relation_name, index_name)
        else:
            condition_seq_filter = []
        condition_seq_index, relation_name, index_name = [], node["Relation Name"], None
        return Scan('Seq Scan', condition_seq_filter, condition_seq_index, relation_name, index_name, cardinality=cardinality, cost=cost, db_estimate_card=db_estimate_card), None
    elif node['Node Type'] == 'Bitmap Heap Scan':
        if 'Filter' in node:
            condition_seq_filter = pre2seq(node['Filter'], alias2table, relation_name, index_name)
        else:
            condition_seq_filter = []
        condition_seq_index, relation_name, index_name = [], node["Relation Name"], None
        return Scan('Bitmap Heap Scan', condition_seq_filter, condition_seq_index, relation_name, index_name, cardinality=cardinality, cost=cost, db_estimate_card=db_estimate_card), None
    elif node['Node Type'] == 'Index Scan':
        if 'Filter' in node:
            condition_seq_filter = pre2seq(node['Filter'], alias2table, relation_name, index_name)
        else:
            condition_seq_filter = []
        if 'Index Cond' in node:
            condition_seq_index = pre2seq(node['Index Cond'], alias2table, relation_name, index_name)
        else:
            condition_seq_index = []
        relation_name, index_name = node["Relation Name"], node['Index Name']
        if len(condition_seq_index) == 1 and re.match(r'[a-zA-Z]+', condition_seq_index[0].right_value) != None:
            return Scan('Index Scan', condition_seq_filter, condition_seq_index, relation_name, index_name, cardinality=cardinality, cost=cost, db_estimate_card=db_estimate_card), condition_seq_index
        else:
            return Scan('Index Scan', condition_seq_filter, condition_seq_index, relation_name, index_name, cardinality=cardinality, cost=cost, db_estimate_card=db_estimate_card), None
    elif node['Node Type'] == 'Bitmap Index Scan':
        if 'Index Cond' in node:
            condition_seq_index = pre2seq(node['Index Cond'], alias2table, relation_name, index_name)
        else:
            condition_seq_index = []
        condition_seq_filter, relation_name, index_name = [], None, node['Index Name']
        if len(condition_seq_index) == 1 and re.match(r'[a-zA-Z]+', condition_seq_index[0].right_value) != None:
            return Scan('Bitmap Index Scan', condition_seq_filter, condition_seq_index, relation_name, index_name, cardinality=cardinality, cost=cost, db_estimate_card=db_estimate_card), condition_seq_index
        else:
            return Scan('Bitmap Index Scan', condition_seq_filter, condition_seq_index, relation_name, index_name, cardinality=cardinality, cost=cost, db_estimate_card=db_estimate_card), None
    elif node['Node Type'] == 'Index Only Scan':
        if 'Index Cond' in node:
            condition_seq_index = pre2seq(node['Index Cond'], alias2table, relation_name, index_name)
        else:
            condition_seq_index = []
        condition_seq_filter, relation_name, index_name = [], None, node['Index Name']
        if len(condition_seq_index) == 1 and re.match(r'[a-zA-Z]+', condition_seq_index[0].right_value) != None:
            return Scan('Index Only Scan', condition_seq_filter, condition_seq_index, relation_name, index_name, cardinality=cardinality, cost=cost, db_estimate_card=db_estimate_card), condition_seq_index
        else:
            return Scan('Index Only Scan', condition_seq_filter, condition_seq_index, relation_name, index_name, cardinality=cardinality, cost=cost, db_estimate_card=db_estimate_card), None

    elif node['Node Type'] == 'Gather' or node['Node Type'] == 'Gather Merge':
        return Gather(node['Workers Planned'], cardinality=cardinality, cost=cost, db_estimate_card=db_estimate_card), None
    
    else:
        raise Exception('Unsupported Node Type: '+node['Node Type'])
        return None, None


def convert_plan_to_sequence(root, alias2table):
    sequence = []
    join_conditions = []
    node, join_condition = encode_node(root, alias2table)
    if join_condition != None:
        join_conditions += join_condition
    sequence.append(node)
    if 'Plans' in root:
        for plan in root['Plans']:
            next_sequence, next_join_conditions = convert_plan_to_sequence(plan, alias2table)
            sequence += next_sequence
            join_conditions += next_join_conditions
    sequence.append(None)
    return sequence, join_conditions
    