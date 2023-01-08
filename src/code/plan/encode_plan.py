import json
import argparse
import os
from .node_features import convert_plan_to_sequence
from .utils import class2json

from ..constants import DATA_ROOT

def get_subplan(root):
    results = []
    if root.has_key('Actual Rows') and root.has_key('Actual Total Time') and root['Actual Rows'] > 0:
        results.append((root, root['Actual Total Time'], root['Actual Rows']))
    if root.has_key('Plans'):
        for plan in root['Plans']:
            results += get_subplan(plan)
    return results

def get_plan(root):
    return (root, root['Actual Total Time'], root['Actual Rows'])

class PlanInSeq(object):
    def __init__(self, seq, cost, cardinality):
        self.seq = seq
        self.cost = cost
        self.cardinality = cardinality
        
def get_alias2table(root, alias2table):
    if 'Relation Name' in root and 'Alias' in root:
        alias2table[root['Alias']] = root['Relation Name']
    if 'Plans' in root:
        for child in root['Plans']:
            get_alias2table(child, alias2table)


def encode_plan(input_path, out_path):
    with open(out_path, 'w') as out:
        with open(input_path) as f:
            # for _, plan in enumerate(f.readlines()):
            #     print(len(plan))
            #     if plan != 'null\n':
            #         plan = json.loads(plan)['Plan']
            #         alias2table = {}
            #         get_alias2table(plan, alias2table)
            #         subplan, cost, cardinality = get_plan(plan)
            #         seq, _ = convert_plan_to_sequence(subplan, alias2table)
            #         seqs = PlanInSeq(seq, cost, cardinality)
            #         out.write(class2json(seqs)+'\n')
            plans = json.load(f)

            for index, plan in enumerate(plans):
                alias2table = {}
                get_alias2table(plan, alias2table)
                root, cost, cardinality = get_plan(plan)
                seq, _ = convert_plan_to_sequence(root, alias2table)
                seqs = PlanInSeq(seq, cost, cardinality)
                out.write(class2json(seqs)+'\n')


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

    for phase in ['train', 'valid', 'test']:
        input_path = os.path.join(DATA_ROOT, dataset, "workload", f"{name}_plans_{phase}.json")
        output_path = os.path.join(DATA_ROOT, dataset, "workload", f"{name}_plans_{phase}_encoded.json")

        encode_plan(input_path=input_path, out_path=output_path)