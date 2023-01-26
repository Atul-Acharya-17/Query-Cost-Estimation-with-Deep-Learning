import os
import argparse
import json

import psycopg2

from ..constants import DATA_ROOT, DATABASE_URL, NUM_TRAIN, NUM_VAL, NUM_TEST, JOB_TRAIN, JOB_LIGHT, SCALE, SYNTHETIC

count = {
    "train": NUM_TRAIN,
    "valid": NUM_VAL,
    "test": NUM_TEST,
    "job-train": JOB_TRAIN,
    "job-light": JOB_LIGHT,
    "scale": SCALE,
    "synthetic": SYNTHETIC
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


def get_execution_plans(dataset, phases=['train', 'valid', 'test']):

    postgres = Postgres()

    query_path = DATA_ROOT / dataset / "workload" / "queries"

    query_plans = {}

    for phase in phases:
        plans = []
        with open(query_path / f"{phase}.sql") as sql_file:
            for index, query in enumerate(sql_file):
                result = postgres.get_plan(query)
                execution_plan = result[0][0][0]['Plan']
                plans.append(execution_plan)
                print(f"{phase} {index+1} / {count[phase]}", end='\r')
        print()
        query_plans[phase] = plans

    dump_plans(dataset=dataset, query_plans=query_plans)


def dump_plans(dataset:str, query_plans:dict):

    plan_path = DATA_ROOT / dataset / "workload" / "plans"
    plan_path.mkdir(exist_ok=True)

    for phase, plans in query_plans.items():
        with open(plan_path / f"{phase}_plans.json", "w") as json_file:
            json.dump(plans, json_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='random')
    parser.add_argument('--version', default='original')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    dataset = args.dataset

    phases = ['train', 'valid', 'test']

    if dataset == 'imdb':
        phases = ['job-train', 'job-light', 'synthetic', 'scale']


    get_execution_plans(dataset, phases)

    