import json
import pandas as pd
import matplotlib.pyplot as plt
from ..constants import RESULT_ROOT

plt.style.use('seaborn-whitegrid')

tree_nn = str(RESULT_ROOT) + '/output/imdb/results_best_tree_nn_job-light_plan.csv'
tree_lstm = str(RESULT_ROOT) + '/output/imdb/results_best_tree_lstm_job-light_plan.csv'
tree_attn = str(RESULT_ROOT) + '/output/imdb/results_best_tree_attn_job-light_plan.csv'
tree_gru = str(RESULT_ROOT) + '/output/imdb/results_best_tree_gru_job-light_plan.csv'

tree_xgb = str(RESULT_ROOT) + '/output/imdb/results_xgb_5_100_16_0.1_fast_False_job-light_plan.csv'
tree_lgbm = str(RESULT_ROOT) + '/output/imdb/results_lgbm_5_100_8_0.1_fast_False_job-light_plan.csv'
tree_xgb_optim = str(RESULT_ROOT) + '/output/imdb/results_xgb_5_100_16_0.1_fast_True_job-light_plan.csv'
tree_lgbm_optim = str(RESULT_ROOT) + '/output/imdb/results_lgbm_5_100_8_0.1_fast_True_job-light_plan.csv'

files = {
    'TreeXGB': tree_xgb,
    'TreeXGB-Optim': tree_xgb_optim,
    'TreeLGBM': tree_lgbm,
    'TreeLGBM-Optim': tree_lgbm_optim
}


if __name__ == '__main__':
    inference_times = []

    for name, file in files.items():
        df = pd.read_csv(file)
        inf_time = df['inference_time']
        inference_times.append(inf_time)


    plt.boxplot(inference_times, labels=list(files.keys()))
    plt.ylabel('Time in seconds')
    plt.show()