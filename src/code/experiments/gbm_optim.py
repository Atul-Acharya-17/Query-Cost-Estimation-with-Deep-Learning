import json
import pandas as pd
import matplotlib.pyplot as plt
from ..constants import RESULT_ROOT
import numpy as np

plt.style.use('seaborn-whitegrid')

tree_xgb = str(RESULT_ROOT) + '/output/imdb/results_tree_xgb_use_estimator_fast_False_job-light_plan.csv'
tree_lgbm = str(RESULT_ROOT) + '/output/imdb/results_tree_lgbm_use_estimator_fast_False_job-light_plan.csv'
tree_xgb_optim = str(RESULT_ROOT) + '/output/imdb/results_tree_xgb_use_estimator_fast_True_job-light_plan.csv'
tree_lgbm_optim = str(RESULT_ROOT) + '/output/imdb/results_tree_lgbm_use_estimator_fast_True_job-light_plan.csv'

files = {
    'TreeXGB': tree_xgb,
    'TreeXGB-Optim': tree_xgb_optim,
    'TreeLGBM': tree_lgbm,
    'TreeLGBM-Optim': tree_lgbm_optim
}


if __name__ == '__main__':
    inference_times = []

    times = {}

    for name, file in files.items():
        df = pd.read_csv(file)
        inf_time = df['inference_time']
        inference_times.append(inf_time)

        times[name] = round(np.mean(inf_time)*1000, 2)
    
    for name in times.keys():
        print(f"{name}", end=' & ')

    print()
    
    for value in times.values():
        print(f"{value}", end=' & ')

    print()



    plt.boxplot(inference_times, labels=list(files.keys()))
    plt.ylabel('Time in seconds')
    plt.show()