import argparse
import matplotlib.pyplot as plt
from ..constants import RESULT_ROOT
import pandas as pd

plt.style.use('seaborn-whitegrid')


files = {
    'TreeNN': 'results_tree_nn',
    'TreeGRU': 'results_tree_gru',
    'TreeLSTM': 'results_tree_lstm',
    'TreeAttn': 'results_tree_attn',
    'TreeXGB': 'results_tree_xgb',
    'TreeLGBM': 'results_tree_lgbm'
}

sizes = ['1000', '2000', '5000', '10000', '20000', '30000', '40000', '50000', '60000', '70000', '80000', '90000', '100000']


if __name__ == '__main__':

    phases = ['synthetic_plan', 'job-light_plan']

    x_data = [int(size) for size in sizes]

    for phase in phases:

        for name, file in files.items():

            y_data = []

            for size in sizes:

                req_file = str(RESULT_ROOT) + '/output/imdb/' + file + '_' + size + '_'

                if name == 'TreeXGB' or name == 'TreeLGBM':
                    req_file += 'use_estimator_fast_True_'

                req_file += phase + '.csv'

                df = pd.read_csv(req_file)

                mean_error = df['cost_errors'].mean()

                y_data.append(mean_error)

            plt.plot(x_data, y_data, label=name, marker='x')

        plt.ylabel('Cost Errors Mean')
        plt.xlabel('Num Training samples')
        plt.legend(list(files.keys()), loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

        plt.show()



                


