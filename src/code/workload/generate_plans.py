import os
import argparse
import json

import psycopg2

from ..constants import DATA_ROOT, DATABASE_URL, NUM_TRAIN, NUM_VAL, NUM_TEST, JOB_TRAIN, JOB_LIGHT, SCALE, SYNTHETIC, SYNTHETIC_500

count = {
    "train": NUM_TRAIN,
    "valid": NUM_VAL,
    "test": NUM_TEST,
    "job-train": JOB_TRAIN,
    "job-light": JOB_LIGHT,
    "scale": SCALE,
    "synthetic": SYNTHETIC,
    'synthetic-500': SYNTHETIC_500
}


class Postgres():

    def __init__(self):

        self.conn = psycopg2.connect(DATABASE_URL)
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()

    def get_plan(self, query):
        sql = f'explain (analyze, format json) {query}'
        self.cursor.execute(sql)

        res = self.cursor.fetchall()

        return res
    
    def disable_parallel(self):
        sql = 'SET max_parallel_workers_per_gather = 0;'
        self.cursor.execute(sql)
        
    def execute_query(self, query):
        self.cursor.execute(query)
        

def get_execution_plans(dataset, phases=['train', 'valid', 'test'], calibrate=False):

    postgres = Postgres()

    query_path = DATA_ROOT / dataset / "workload" / "queries"

    query_plans = {}
    
    postgres.disable_parallel()
    
    if calibrate:
        with open(DATA_ROOT / "calibration.sql", "r") as f:
            queries = f.readlines()
            for query in queries:
                postgres.execute_query(query)

    for phase in phases:
        plans = []
        with open(query_path / f"{phase}.sql") as sql_file:
            for index, query in enumerate(sql_file):
                result = postgres.get_plan(query)
                execution_plan = result[0][0][0]['Plan']
                plans.append(execution_plan)
                print(f"{phase} {index+1} / {len(sql_file)}", end='\r')
        print()
        query_plans[phase] = plans

    dump_plans(dataset=dataset, query_plans=query_plans)


def dump_plans(dataset:str, query_plans:dict):

    plan_path = DATA_ROOT / dataset / "workload" / "plans"
    plan_path.mkdir(exist_ok=True)

    for phase, plans in query_plans.items():
        with open(plan_path / f"{phase}_plans_calib.json", "w") as json_file:
            json.dump(plans, json_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='random')
    parser.add_argument('--version', default='original')
    parser.add_argument('--calibration', default='False')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    dataset = args.dataset
    calibration = args.calibration
    
    calibrate = False
    
    if calibration == "True":
        calibrate = True

    phases = ['train', 'valid', 'test']

    if dataset == 'imdb':
        #phases = ['job-train', 'job-light', 'synthetic', 'scale']
        phases = ['synthetic-500', 'job-light']


    get_execution_plans(dataset, phases, calibrate)
