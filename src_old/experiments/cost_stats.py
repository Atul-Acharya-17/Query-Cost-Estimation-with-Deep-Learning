import pandas as pd
import numpy as np


CENSUS = 'census_original-base-postgres-version=original;stat=10000;seed=123.csv'
FOREST = 'forest_original-base-postgres-version=original;stat=10000;seed=123.csv'
POWER = 'power_original-base-postgres-version=original;stat=10000;seed=123.csv'
DMV = 'dmv_original-base-postgres-version=original;stat=10000;seed=123.csv'


statistics = {
    'census': CENSUS,
    'forest': FOREST,
    'power': POWER,
    'dmv:': DMV
}

if __name__ == '__main__':

    columns = ['planning_time', 'execution_time', 'total_time']

    for name, filename in statistics.items():
        df = pd.read_csv(f'data/{filename}')

        for col in columns:
            req_stat = df[col]

            stats = {
                'mean': np.mean(req_stat),
                '50th': np.percentile(req_stat, 50),
                '95th': np.percentile(req_stat, 95),
                '99th': np.percentile(req_stat, 99),
                'Max': np.max(req_stat)
            }

            print(f'{name}, {col}')
            print(stats, end='\n*****************************\n')


