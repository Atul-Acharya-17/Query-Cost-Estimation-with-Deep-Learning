import json
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import time
import pandas as pd

import psycopg2

from .constants import DATA_ROOT, RESULT_ROOT, DATABASE_URL
from .train.loss_fn import q_error


def plot_linreg(data, actual, pred):
    plt.scatter(data, actual, color="#242424", s=5)
    plt.plot(data, pred, color="blue")
    plt.xlabel("Total Cost")
    plt.ylabel("Total Time (ms)")
    plt.show()

def card_errors(file_paths):

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            plans = json.load(f)
            card_losses = []
            actual = []
            preds = []

            for index, plan in enumerate(plans):
                print(f"{index+1}/{len(plans)}" ,end='\r')

                estimated_card = plan['Plan Rows']
                actual_card = plan['Actual Rows'] 

                card_error = q_error(estimated_card, actual_card)

                card_losses.append(card_error)
                actual.append(actual_card)
                preds.append(estimated_card)

            print(len(card_losses))
            metrics = {
                'max': np.max(card_losses),
                '99th': np.percentile(card_losses, 99),
                '95th': np.percentile(card_losses, 95),
                '90th': np.percentile(card_losses, 90),
                'median': np.median(card_losses),
                'mean': np.mean(card_losses),
                'MAE': np.mean([abs(preds[i] - actual[i]) for i in range(min(len(preds), len(actual)))])
            }
            
            print(f"Postgres & {round(metrics['mean'], 2)} & {round(metrics['median'], 2)} & {round(metrics['90th'], 2)} & {round(metrics['95th'], 2)} & {round(metrics['99th'], 2)} & {round(metrics['max'], 2)} & {round(metrics['MAE'], 2)}")


def cost_errors(train_path, test_paths):
    model = LinearRegression()
    with open(train_path, 'r') as f:
        plans = json.load(f)
        x_train = []
        y_train = []
        for _, plan in enumerate(plans):
            cost = plan['Total Cost']
            time = plan['Actual Total Time']

            x_train.append([cost])
            y_train.append([time])

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        model.fit(x_train, y_train)
    
    for file in test_paths:

        with open(file, 'r') as f:
            cost_losses = []

            plans = json.load(f)
            actual = []
            preds = []
            data = []

            for _, plan in enumerate(plans):
                cost = plan['Total Cost']
                time = plan['Actual Total Time']

                data.append(cost)
                actual.append(time)

                estimated_time = model.predict(np.array([[cost]]))[0]

                preds.append(estimated_time)

                cost_error = q_error(estimated_time, time)

                cost_losses.append(cost_error)

            print(len(cost_losses))
            metrics = {
                'max': np.max(cost_losses),
                '99th': np.percentile(cost_losses, 99),
                '95th': np.percentile(cost_losses, 95),
                '90th': np.percentile(cost_losses, 90),
                'median': np.median(cost_losses),
                'mean': np.mean(cost_losses),
                'MAE': np.mean([abs(preds[i] - actual[i]) for i in range(min(len(preds), len(actual)))])
            }

            print(f"Postgres & {round(metrics['mean'], 2)} & {round(metrics['median'], 2)} & {round(metrics['90th'], 2)} & {round(metrics['95th'], 2)} & {round(metrics['99th'], 2)} & {round(metrics['max'], 2)} & {round(metrics['MAE'], 2)}")
            plot_linreg(data, actual, preds)


def inference_time(filenames):
    conn = psycopg2.connect(DATABASE_URL)
    conn.autocommit = True
    cursor = conn.cursor()

    for file in filenames:

        inference_times = []

        with open(file) as sql_file:
            for _, query in enumerate(sql_file):
                sql = f'explain {query}'
                start_time = time.time()
                cursor.execute(sql)
                end_time = time.time()
                res = cursor.fetchall()
                inference_times.append((end_time - start_time))

        metrics = {
            'max': np.max(inference_times),
            '99th': np.percentile(inference_times, 99),
            '95th': np.percentile(inference_times, 95),
            '90th': np.percentile(inference_times, 90),
            'median': np.median(inference_times),
            'mean': np.mean(inference_times),        
        }

        print(f"{round(metrics['mean'] * 1000, 2)} & {round(metrics['median'] * 1000, 2)} & {round(metrics['90th'] * 1000, 2)} & {round(metrics['95th'] * 1000, 2)} & {round(metrics['99th'] * 1000, 2)} & {round(metrics['max'] * 1000, 2)}")

        stats_df = pd.DataFrame(list(zip(inference_times)), columns=['inference_time'])
        stats_df.to_csv(str(RESULT_ROOT) + "/output/imdb" + f"/results_postgres_job_light_plans.csv")



if __name__ == '__main__':
    train_plans = str(DATA_ROOT) + '/imdb/workload/plans/train_plan_100000.json'
    synthetic_plans = str(DATA_ROOT) + '/imdb/workload/plans/synthetic_plan.json'
    job_light_plans = str(DATA_ROOT) + '/imdb/workload/plans/job-light_plan.json'

    job_light_queries = str(DATA_ROOT) + '/imdb/workload/queries/job-light.sql'

    print('Cardinality')
    card_errors([train_plans, synthetic_plans, job_light_plans])

    print('\nCost')

    cost_errors(train_plans, [train_plans, synthetic_plans, job_light_plans])

    print('\nInference Time (ms)')
    inference_time([job_light_queries])