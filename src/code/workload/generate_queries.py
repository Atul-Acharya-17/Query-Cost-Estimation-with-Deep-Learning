import random
import logging
import numpy as np
from typing import Dict, Any
import copy

from . import generator
from .generator import QueryGenerator
from .workload import dump_queryset, query_2_sql, dump_querystring
from ..dataset.dataset import load_table, Table, Column

import argparse

L = logging.getLogger(__name__)

def get_focused_table(table, ref_table, win_ratio):
    focused_table = copy.deepcopy(table)
    win_size = int(win_ratio * len(ref_table.data))
    focused_table.data = focused_table.data.tail(win_size).reset_index(drop=True)
    focused_table.parse_columns()
    return focused_table

def generate_queries(
    seed: int, dataset: str, version: str,
    name: str,
    params: Dict[str, Dict[str, Any]]
) -> None:

    random.seed(seed)
    np.random.seed(seed)

    attr_funcs = {getattr(generator, f"asf_{a}"): v for a, v in params['attr'].items()}
    center_funcs = {getattr(generator, f"csf_{c}"): v for c, v in params['center'].items()}
    width_funcs = {getattr(generator, f"wsf_{w}"): v for w, v in params['width'].items()}

    print("Load table...")
    table = load_table(dataset, version)

    qgen = QueryGenerator(
        table=table,
        attr=attr_funcs,
        center=center_funcs,
        width=width_funcs,
        attr_params=params.get('attr_params') or {},
        center_params=params.get('center_params') or {},
        width_params=params.get('width_params') or {})

    queryset = {}
    query_strings = {}
    for group, num in params['number'].items():
        print(f"Start generate workload with {num} queries for {group}...")
        queries = []
        strings = []
        for i in range(num):
            query = qgen.generate()
            query_string = query_2_sql(query=query, table=table)
            strings.append(query_string)
            queries.append(query)
            if (i+1) % 1000 == 0:
                print(f"{i+1} queries generated")
        queryset[group] = queries
        query_strings[group] = strings

    print("Dump queryset to disk...")
    dump_queryset(dataset, name, queryset)

    # Save every query in sql file
    dump_querystring(dataset, name, query_strings)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='random')
    parser.add_argument('--version', default='original')
    parser.add_argument('--name', default='base')
    parser.add_argument('--seed', default=2023)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()


    dataset = args.dataset
    version = args.version
    name = args.name
    seed = args.seed

    params = {'attr': {'pred_number': 1.0}, \
            'center': {'distribution': 0.9, 'vocab_ood': 0.1}, \
            'width': {'uniform': 0.5, 'exponential': 0.5}, \
            'number': {'train': 1000, 'valid': 100, 'test': 100}
        }

    generate_queries(seed=seed, dataset=dataset, version=version, name=name, params=params)