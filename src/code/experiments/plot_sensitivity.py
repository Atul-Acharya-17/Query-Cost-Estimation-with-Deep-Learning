import argparse
import matplotlib.pyplot as plt
from ..constants import RESULT_ROOT
import pandas as pd

plt.style.use('seaborn-whitegrid')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='xgb')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    name = args.name

    num_train = ['1000', '2000', '5000', '10000', '20000', '30000', '40000', '50000', '60000','70000', '80000', '90000', '100000']
    num_models = ['1', '1', '1', '1', '1', '2', '3', '3', '4', '4', '4', '5', '5', '5']

    job_light_files = []
    synthetic_files = []

    for idx, num in enumerate(num_train):
        file = str(RESULT_ROOT) + '/output/imdb/results_' +  '_tree_' + name + '_' + num + f'_fast_True_synthetic_plan.csv'
        synthetic_files.append(file)
        file = str(RESULT_ROOT) + '/output/imdb/results_' + name +  '_tree_' + name + '_' + num + f'_fast_True_job-light_plan.csv'
        job_light_files.append(file)

    
    x_data = []

    y_data_synthetic_mean = []
    y_data_synthetic_median = []

    y_data_job_light_mean = []
    y_data_job_light_median = []


    for num in num_train:
        x_data.append(int(num))

    for file in synthetic_files:
        df = pd.read_csv(file)

        mean_error = df['cost_errors'].mean()

        median_error = df['cost_errors'].median()

        y_data_synthetic_mean.append(mean_error)
        y_data_synthetic_median.append(median_error)

    for file in job_light_files:
        df = pd.read_csv(file)

        mean_error = df['cost_errors'].mean()

        median_error = df['cost_errors'].median()

        y_data_job_light_mean.append(mean_error)
        y_data_job_light_median.append(median_error)

    plt.plot(x_data, y_data_synthetic_mean, label='Synthetic', marker='o')
    plt.plot(x_data, y_data_job_light_mean, label='JOB-ligth', marker='o')
    plt.ylabel('Cost Errors Mean')
    plt.xlabel('Num Training samples')
    plt.legend(['Synthetic500', 'JOB-light'])

    plt.show()
    
    plt.plot(x_data, y_data_synthetic_median, label='Synthetic', marker='o')
    plt.plot(x_data, y_data_job_light_median, label='JOB-light', marker='o')
    plt.ylabel('Cost Errors Median')
    plt.xlabel('Num Training samples')
    plt.legend(['Synthetic500', 'JOB-light'])

    plt.show()
    