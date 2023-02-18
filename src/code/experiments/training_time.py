import json
import pandas as pd
import matplotlib.pyplot as plt
from ..constants import RESULT_ROOT


plt.style.use('seaborn-whitegrid')


tree_nn = str(RESULT_ROOT) + '/output/imdb/training_statistics_tree_nn_train.json'
tree_lstm = str(RESULT_ROOT) + '/output/imdb/training_statistics_tree_lstm_train.json'
tree_attn = str(RESULT_ROOT) + '/output/imdb/training_statistics_tree_attn_train.json'
tree_gru = str(RESULT_ROOT) + '/output/imdb/training_statistics_tree_gru_train.json'

tree_xgb = str(RESULT_ROOT) + '/output/imdb/training_statistics_xgb_tree_xgb_train_plan_100000.json'
tree_lgbm = str(RESULT_ROOT) + '/output/imdb/training_statistics_lgbm_tree_lgbm_train_plan_100000.json'


files = {
    'TreeNN':tree_nn,
    'TreeGRU': tree_gru,
    'TreeLSTM': tree_lstm,
    'TreeAttn': tree_attn,
    'TreeXGB': tree_xgb,
    'TreeLGBM': tree_lgbm
}

if __name__ =='__main__':

    times = {

    }

    for name, file in files.items():
        with open(file, 'r') as f:
            data = json.load(f)

            training_times = data['training_times']
            total_time = sum(training_times)
            mean = total_time / len(training_times)

            times[name] = {
                'mean': mean,
                'total': total_time,
                'count': len(training_times)
            }

    for k, v in times.items():
        print(f"{k} & {v['count']} & {round(v['mean'], 2)} & {round(v['total'], 2)} \\\\")
