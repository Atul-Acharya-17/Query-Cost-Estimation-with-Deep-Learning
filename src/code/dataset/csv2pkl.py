from pathlib import Path
import pandas as pd
from pandas.api.types import CategoricalDtype
from ..constants import DATA_ROOT
import os
import argparse

from .dataset import load_table


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='random')
    parser.add_argument('--version', default='original')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()
    dataset = args.dataset
    version = args.version

    path = os.path.join(DATA_ROOT, dataset)
    csv_path = os.path.join(path, version) + ".csv"
    pkl_path = os.path.join(path, version) + ".pkl"
    df = pd.read_csv(csv_path)
    df = df.astype({k: CategoricalDtype(ordered=True) for k, d in df.dtypes.items() if d == "O"})
    df.to_pickle(pkl_path)
    load_table(dataset, version)