import json
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plans-file', default='plans.json')
    args = parser.parse_args()
    return args

dict = {}

def traverse_tree(plan, nodes, depth):
    nodes += 1
    
    if plan["Node Type"] in dict:
        dict[plan["Node Type"]] += 1 
    else:
        dict[plan["Node Type"]] = 1
    if 'Plans' not in plan or len(plan['Plans']) == 0:
        return nodes, 0
    
    depths = []
    for child in plan['Plans']:
        n, d = traverse_tree(child, nodes, depth)
        nodes = n
        depths.append(d)
        
    return nodes, max(depths) + 1
    

def extract_stats(plan_file):
    
    max_depth = 0
    depth_sum = 0
    
    max_nodes = 0
    nodes_sum = 0
    
    num_items = 0
    
    with open(plan_file, 'r') as f:
        plans = json.load(f)
        num_items = len(plans)
        for idx, plan in enumerate(plans): 
            
            print(f'{idx + 1}/{len(plans)}')
            
            nodes, depth = traverse_tree(plan, 0, 0)

            max_nodes = max(max_nodes, nodes)
            max_depth = max(max_depth, depth)
            
            depth_sum += depth
            nodes_sum += nodes
            
    print(f'Max Nodes: {max_nodes}')
    print(f'Max Depth: {max_depth}')
    
    print(f'Avg Nodes: {nodes_sum / num_items}')
    print(f'Avg Depth: {depth_sum / num_items}')

    print(nodes_sum, num_items)
            

if __name__ == '__main__':
    args = parse_args()
    
    plans_file = args.plans_file
    
    extract_stats(plans_file)

    print(dict)
    
    
    