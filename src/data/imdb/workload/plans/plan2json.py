import pandas as pd
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv-file', default='train_plans.csv')
    parser.add_argument('--json-file', default='train_plans.json')
    args = parser.parse_args()
    return args


def convert_csv_plan_to_json(file='', save_file=''):
    
    plans = []
    
    df = pd.read_csv(file)
    
    for idx in range(len(df)):
        plan = df.iloc[idx]['json']
        plan = json.loads(plan)['Plan']
        plans.append(plan)
        
    with open(save_file, 'w') as f:
        json.dump(plans, f)
    
if __name__ == '__main__':
    args = parse_args()
    
    file = args.csv_file
    save_file = args.json_file
    
    convert_csv_plan_to_json(file, save_file)