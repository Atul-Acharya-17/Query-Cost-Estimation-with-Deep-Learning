import os
import argparse
import json

import psycopg2

from ..constants import DATA_ROOT, DATABASE_URL


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


def get_execution_plans(dataset, version, name, phases=['train', 'valid', 'test']):

    postgres = Postgres()

    query_path = DATA_ROOT / dataset / "workload"

    query_plans = {}

    for phase in phases:
        plans = []
        with open(query_path / f"{name}_{phase}.sql") as sql_file:
            for query in sql_file:
                result = postgres.get_plan(query)
                execution_plan = result[0][0][0]['Plan']
                plans.append(execution_plan)
            
        query_plans[phase] = plans

    dump_plans(dataset=dataset, version=version, name=name, query_plans=query_plans)


def dump_plans(dataset:str, version:str, name:str, query_plans:dict):

    query_path = DATA_ROOT / dataset / "workload"

    for phase, plans in query_plans.items():
        with open(query_path / f"{name}_plans_{phase}.json", "w") as json_file:
            json.dump(plans, json_file)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='random')
    parser.add_argument('--version', default='original')
    parser.add_argument('--name', default='base')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    dataset = args.dataset
    version = args.version
    name = args.name

    phases = ['train', 'valid', 'test']

    get_execution_plans(dataset, version, name, phases)

    