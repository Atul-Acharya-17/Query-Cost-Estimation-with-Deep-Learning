import numpy as np
import pandas as pd
from ..constants import RESULT_ROOT


tree_nn = str(RESULT_ROOT) + '/output/imdb/results_best_tree_nn'
tree_lstm = str(RESULT_ROOT) + '/output/imdb/results_best_tree_lstm'
tree_attn = str(RESULT_ROOT) + '/output/imdb/results_best_tree_attn'
tree_gru = str(RESULT_ROOT) + '/output/imdb/results_best_tree_gru'

tree_xgb = str(RESULT_ROOT) + '/output/imdb/results_tree_xgb_use_estimator_fast_False'
tree_lgbm = str(RESULT_ROOT) + '/output/imdb/results_tree_lgbm_use_estimator_fast_False'


files = {
    'TreeNN':tree_nn,
    'TreeGRU': tree_gru,
    'TreeLSTM': tree_lstm,
    'TreeAttn': tree_attn,
    'TreeXGB': tree_xgb,
    'TreeLGBM': tree_lgbm
}


if __name__ == '__main__':
    inference_times = []

    for error in ['card', 'cost']:

        for phase in ['synthetic_plan', 'job-light_plan']:
            
            print('-'*100)
            print(f'{phase} - {error}')
            print('-'*100)

            for name, file in files.items():
                
                df = pd.read_csv(file + '_' + phase + '.csv')
                errors = df[error + '_errors']

                stats = {
                    'max': np.max(errors),
                    '99th': np.percentile(errors, 99),
                    '95th': np.percentile(errors, 95),
                    '90th': np.percentile(errors, 90),
                    'median': np.median(errors),
                    'mean': np.mean(errors),
                    'mae': np.mean(abs((df[error + '_pred'] - df[error + '_actual'])))
                }

                print(f"{name} & {round(stats['mean'], 2)} & {round(stats['median'], 2)} & {round(stats['90th'], 2)} & {round(stats['95th'], 2)} & {round(stats['99th'], 2)} & {round(stats['max'], 2)} & {round(stats['mae'], 2)}")
                
