import argparse
import pandas as pd

from ..constants import RESULT_ROOT

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='imdb')
    parser.add_argument('--name', default='TreeLSTM')
    parser.add_argument('--file')


    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_args()

    dataset = args.dataset
    name = args.name
    file = args.file

    directory = str(RESULT_ROOT) + '/output/'

    df = pd.read_csv(directory + dataset + '/' +  file)
    print('Cost')
    print(f'{name} & {round(df.cost_errors.mean(), 2)} & {round(df.cost_errors.median(), 2)} & {round(df.cost_errors.quantile(0.9), 2)} & {round(df.cost_errors.quantile(0.95), 2)} & {round(df.cost_errors.quantile(0.99), 2)} & {round(df.cost_errors.max(), 2)} \\\\')
    print('-'*50)
    print('Cardinality')
    print(f'{name} & {round(df.card_errors.mean(), 2)} & {round(df.card_errors.median(), 2)} & {round(df.card_errors.quantile(0.9), 2)} & {round(df.card_errors.quantile(0.95), 2)} & {round(df.card_errors.quantile(0.99), 2)} & {round(df.card_errors.max(), 2)} \\\\')
    print('-'*50)
    print('Inference Time')
    print(f'{name} & {df.inference_time.mean()} & {df.inference_time.median()} & {df.inference_time.quantile(0.9)} & {df.inference_time.quantile(0.95)} & {df.inference_time.quantile(0.99)} & {df.inference_time.max()} \\\\')
    print('-'*50)