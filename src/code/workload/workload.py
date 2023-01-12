import csv
from collections import OrderedDict
from typing import Dict, NamedTuple, Optional, Tuple, List, Any
import pickle
import numpy as np
import os

from ..dtypes import is_categorical
from ..constants import DATA_ROOT, PKL_PROTO
from ..dataset.dataset import Table, load_table


class Query(NamedTuple):
    """predicate of each attritbute are conjunctive"""
    predicates: Dict[str, Optional[Tuple[str, Any]]]
    ncols: int

class Label(NamedTuple):
    cardinality: int
    selectivity: float

def new_query(table: Table, ncols) -> Query:
    return Query(predicates=OrderedDict.fromkeys(table.data.columns, None),
                 ncols=ncols)

def query_2_triple(query: Query, with_none: bool=True, split_range: bool=False
               ) -> Tuple[List[int], List[str], List[Any]]:
    """return 3 lists with same length: cols(columns names), ops(predicate operators), vals(predicate literals)"""
    cols = []
    ops = []
    vals = []
    for c, p in query.predicates.items():
        if p is not None:
            if split_range is True and p[0] == '[]':
                cols.append(c)
                ops.append('>=')
                vals.append(p[1][0])
                cols.append(c)
                ops.append('<=')
                vals.append(p[1][1])
            else:
                cols.append(c)
                ops.append(p[0])
                vals.append(p[1])
        elif with_none:
            cols.append(c)
            ops.append(None)
            vals.append(None)
    return cols, ops, vals

def query_2_sql(query: Query, table: Table, aggregate=True, split=False, dbms='postgres'):
    preds = []
    for col, pred in query.predicates.items():
        if pred is None:
            continue
        op, val = pred
        if is_categorical(table.data[col].dtype):
            val = f"\'{val}\'" if not isinstance(val, tuple) else tuple(f"\'{v}\'" for v in val)
        if op == '[]':
            if split:
                preds.append(f"{col} >= {val[0]}")
                preds.append(f"{col} <= {val[1]}")
            else:
                preds.append(f"({col} between {val[0]} and {val[1]})")
        else:
            preds.append(f"{col} {op} {val}")

    return f"SELECT {'COUNT(*)' if aggregate else '*'} FROM \"{table.name}\" WHERE {' AND '.join(preds)};"

def query_2_sqls(query: Query, table: Table):
    sqls = []
    for col, pred in query.predicates.items():
        if pred is None:
            continue
        op, val = pred
        if is_categorical(table.data[col].dtype):
            val = f"\'{val}\'" if not isinstance(val, tuple) else tuple(f"\'{v}\'" for v in val)

        if op == '[]':
            sqls.append(f"SELECT * FROM \"{table.name}\" WHERE {col} between {val[0]} and {val[1]}")
        else:
            sqls.append(f"SELECT * FROM \"{table.name}\" WHERE {col} {op} {val}")
    return sqls


def dump_queryset(dataset: str, name: str, queryset: Dict[str, List[Query]]) -> None:
    query_path = DATA_ROOT / dataset / "workload" / "queries"
    query_path.mkdir(parents=True, exist_ok=True)
    with open(query_path / f"{name}.pkl", 'wb') as f:
        pickle.dump(queryset, f, protocol=PKL_PROTO)

def dump_querystring(dataset:str, query_strings: Dict[str, List[str]]) -> None:
    query_path = DATA_ROOT / dataset / "workload" / "queries"
    query_path.mkdir(exist_ok=True)

    for group, queries in query_strings.items():
        strings = '\n'.join(queries)
        with open(query_path / f"{group}.sql", 'w') as file:
            file.write(strings)


def load_queryset(dataset: str, name: str) -> Dict[str, List[Query]]:
    query_path = DATA_ROOT / dataset / "workload" / "queries"
    with open(query_path / f"{name}.pkl", 'rb') as f:
        return pickle.load(f)
