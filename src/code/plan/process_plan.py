import json
import argparse
import os

from .node_features import convert_plan_to_sequence
from .utils import class2json
from .bitmap_features import TreeNode, recover_tree, get_bitmap, bitand
from ..constants import DATA_ROOT, SAMPLE_NUM


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


def encode_plan(input_path, out_path, dataset, data=None, samples=None):
    with open(out_path, 'w') as out:
        with open(input_path) as f:
            plans = json.load(f)
            for index, plan in enumerate(plans):
                alias2table = {}
                get_alias2table(plan, alias2table)
                root, cost, cardinality = get_plan(plan)
                seq, _ = convert_plan_to_sequence(root, alias2table)
                seqs = PlanInSeq(seq, cost, cardinality)
                parsed_plan = json.loads(class2json(seqs))
                nodes_with_sample = []
                for node in parsed_plan['seq']:
                    bitmap_filter = []
                    bitmap_index = []
                    bitmap_other = []
                    if node != None and 'condition' in node:
                        predicates = node['condition']
                        if len(predicates) > 0:
                            root = TreeNode(predicates[0], None)
                            if len(predicates) > 1:
                                recover_tree(predicates[1:], root)
                            bitmap_other = get_bitmap(root, dataset, data=data, sample=samples)
                    if node != None and 'condition_filter' in node:
                        predicates = node['condition_filter']
                        if len(predicates) > 0:
                            root = TreeNode(predicates[0], None)
                            if len(predicates) > 1:
                                recover_tree(predicates[1:], root)
                            bitmap_filter = get_bitmap(root, dataset, data=data, sample=samples)
                    if node != None and 'condition_index' in node:
                        predicates = node['condition_index']
                        if len(predicates) > 0:
                            root = TreeNode(predicates[0], None)
                            if len(predicates) > 1:
                                recover_tree(predicates[1:], root)
                            bitmap_index = get_bitmap(root, dataset, data=data, sample=samples)
                    if len(bitmap_filter) > 0 or len(bitmap_index) > 0 or len(bitmap_other) > 0:
                        bitmap = [1 for _ in range(SAMPLE_NUM)]
                        bitmap = bitand(bitmap, bitmap_filter)
                        bitmap = bitand(bitmap, bitmap_index)
                        bitmap = bitand(bitmap, bitmap_other)
                        node['bitmap'] = ''.join([str(x) for x in bitmap])
                    nodes_with_sample.append(node)
                parsed_plan['seq'] = nodes_with_sample

                out.write(json.dumps(parsed_plan))
                out.write('\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='random')
    parser.add_argument('--version', default='original')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    dataset = args.dataset
    version = args.version

    phases = ['train', 'valid', 'test']

    if dataset == 'imdb':
        phases = ['job-light_plan', 'synthetic_plan', 'train_plan_100000']

        from ..dataset.imdb import get_data_and_samples

        data, samples = get_data_and_samples()

        for phase in phases:
            input_path = os.path.join(DATA_ROOT, dataset, "workload/plans", f"{phase}.json")
            output_path = os.path.join(DATA_ROOT, dataset, "workload/plans", f"{phase}_encoded.json")
            encode_plan(input_path=input_path, out_path=output_path, dataset=dataset, data=data, samples=samples)


    else:
        for phase in phases:
            input_path = os.path.join(DATA_ROOT, dataset, "workload/plans", f"{phase}_plans.json")
            output_path = os.path.join(DATA_ROOT, dataset, "workload/plans", f"{phase}_plans_encoded.json")

            encode_plan(input_path=input_path, out_path=output_path, dataset=dataset)