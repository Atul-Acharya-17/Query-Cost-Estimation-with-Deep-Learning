import random
import numpy as np
import pandas as pd
from scipy.stats import truncnorm, truncexpon, genpareto
from typing import Dict, Any
import argparse

from ..constants import DATA_ROOT
from .dataset import load_table, Table, Column


def get_truncated_normal(mean=0, sd=100, low=0, upp=1000):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def get_truncated_expon(scale=100, low=0, upp=1000):
    return truncexpon(b=(upp-low)/scale, loc=low, scale=scale)

def generate_dataset(
    seed: int, dataset: str, version: str,
    params: Dict[str, Any], overwrite: bool
) -> None:
    path = DATA_ROOT / dataset
    path.mkdir(exist_ok=True)
    csv_path = path / f"{version}.csv"
    pkl_path = path / f"{version}.pkl"
    if not overwrite and csv_path.is_file():
        print(f"Dataset path exists, do not continue")
        return

    row_num = params['row_num']
    col_num = params['col_num']
    dom = params['dom']
    corr = params['corr']
    skew = params['skew']

    if col_num != 2:
        print("For now only support col=2!")
        exit(0)

    print(f"Generating dataset with {col_num} columns and {row_num} rows using seed {seed}")
    random.seed(seed)
    np.random.seed(seed)

    # generate the first column according to skew
    col0 = np.arange(dom) # make sure every domain value has at least 1 value
    tmp = genpareto.rvs(skew-1, size=row_num-len(col0)) # c = skew - 1, so we can have c >= 0
    tmp = ((tmp - tmp.min()) / (tmp.max() - tmp.min())) * dom # rescale generated data to the range of domain
    col0 = np.concatenate((col0, np.clip(tmp.astype(int), 0, dom-1)))

    # generate the second column according to the first
    col1 = []
    for c0 in col0:
        col1.append(c0 if np.random.uniform(0, 1) <= corr else np.random.choice(dom))

    df = pd.DataFrame(data={'col0': col0, 'col1': col1})

    print(f"Dump dataset {dataset} as version {version} to disk")
    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)
    load_table(dataset, version)
    print(f"Finish!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='random')
    parser.add_argument('--version', default='original')
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--rows', default=10000, type=int)
    parser.add_argument('--cols', default=2, type=int)
    parser.add_argument('--skew', default=0.0, type=float)
    parser.add_argument('--corr', default=0.0, type=float)
    parser.add_argument('--dom', default=1000, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    dataset = args.dataset
    version = args.version
    seed = args.seed
    rows = args.rows
    cols = args.cols
    skew = args.skew
    corr = args.corr
    dom = args.dom

    params = {
        'row_num': rows,
        'col_num': cols,
        'corr': corr,
        'skew': skew,
        'dom': dom
    }

    generate_dataset(seed=seed, dataset=dataset, version=version, params=params, overwrite=True)