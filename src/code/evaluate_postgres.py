import argparse
import json
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from .constants import DATA_ROOT
from .train.loss_fn import q_error


def plot_linreg(data, actual, pred):
    plt.scatter(data, actual, color="black")
    plt.plot(data, pred, color="blue", linewidth=3)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def evaluate_postgres_card(file_path):
    card_losses = []

    with open(file_path, 'r') as f:
        plans = json.load(f)
        for index, plan in enumerate(plans):
            print(f"{index}/{len(plans)}" ,end='\r')

            # estimated_cost = plan['Total Cost']
            # actual_cost = plan['Actual Total Time']

            estimated_card = plan['Plan Rows']
            actual_card = plan['Actual Rows'] 

            # cost_error = q_error(estimated_cost, actual_cost)
            card_error = q_error(estimated_card, actual_card)

            card_losses.append(card_error)
            # cost_losses.append(cost_error)

    return card_losses


def evaluate_postgres_cost(train_path, test_path):
    model = LinearRegression()
    cost_losses = []
    with open(train_path, 'r') as f:
        plans = json.load(f)
        x_train = []
        y_train = []
        for index, plan in enumerate(plans):
            cost = plan['Total Cost']
            time = plan['Actual Total Time']

            x_train.append([cost])
            y_train.append([time])

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        model.fit(x_train, y_train)

        preds = model.predict(x_train)

        plot_linreg(x_train.flatten(), y_train.flatten(), preds.flatten())

    with open(test_path, 'r') as f:
        plans = json.load(f)
        actual = []
        preds = []
        data = []
        for idx, plan in enumerate(plans):
            cost = plan['Total Cost']
            time = plan['Actual Total Time']

            data.append(cost)
            actual.append(time)

            estimated_time = model.predict(np.array([[cost]]))[0]

            preds.append(estimated_time)

            cost_error = q_error(estimated_time, time)

            cost_losses.append(cost_error)

        plot_linreg(data, actual, preds)

    return cost_losses


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='census13')
    parser.add_argument('--phase', default='test')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    dataset = args.dataset
    phase = args.phase

    file_path = f"{DATA_ROOT}/{dataset}/workload/plans/{phase}_plans.json"

    card_losses = evaluate_postgres_card(file_path)

    train_path = f"{DATA_ROOT}/{dataset}/workload/plans/train_plans.json"
    test_path = f"{DATA_ROOT}/{dataset}/workload/plans/{phase}_plans.json"
    cost_losses = evaluate_postgres_cost(train_path, test_path)

    cost_metrics = {
        'max': np.max(cost_losses),
        '99th': np.percentile(cost_losses, 99),
        '95th': np.percentile(cost_losses, 95),
        '90th': np.percentile(cost_losses, 90),
        'median': np.median(cost_losses),
        'mean': np.mean(cost_losses),
    }

    card_metrics = {
        'max': np.max(card_losses),
        '99th': np.percentile(card_losses, 99),
        '95th': np.percentile(card_losses, 95),
        '90th': np.percentile(card_losses, 90),
        'median': np.median(card_losses),
        'mean': np.mean(card_losses),
    }


    print(f"cost metrics: {cost_metrics}, \ncardinality metrics: {card_metrics}")
