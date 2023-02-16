import json
import pandas as pd
import matplotlib.pyplot as plt
from ..constants import RESULT_ROOT

plt.style.use('seaborn-whitegrid')


tree_lgbm_est_card = str(RESULT_ROOT) + '/output/imdb/results_tree_lgbm_use_estimator_fast_True_job-light_plan.csv'
tree_lgbm_db_card = str(RESULT_ROOT) + '/output/imdb/results_tree_lgbm_use_db_pred_fast_True_job-light_plan.csv'

tree_xgb_est_card = str(RESULT_ROOT) + '/output/imdb/results_tree_xgb_use_estimator_fast_True_job-light_plan.csv'
tree_xgb_db_card = str(RESULT_ROOT) + '/output/imdb/results_tree_xgb_use_db_pred_fast_True_job-light_plan.csv'


files = {
    'TreeLGBM \n est_card': tree_lgbm_est_card,
    'TreeLGBM \n db_card': tree_lgbm_db_card,
    'TreeXGB \n est_card': tree_xgb_est_card,
    'TreeXGB \n db_card': tree_xgb_db_card
    
}


if __name__ == '__main__':
    inference_times = []
    table = {

    }

    for name, file in files.items():
        df = pd.read_csv(file)
        inf_time = df['inference_time']
        inference_times.append(inf_time)

        table[name] = df['inference_time'].mean()

    plt.boxplot(inference_times, labels=list(files.keys()))
    plt.ylabel('Time in seconds')
    plt.show()

    print(' & '.join(table.keys()) + '\\\\')
    print(' & '.join([str(round(1000 * x, 2)) for x in table.values()]) + '\\\\')