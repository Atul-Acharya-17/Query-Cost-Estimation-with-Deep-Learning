import pickle

def get_batch_job_tree(batch_id, phase, directory):
    suffix = phase + "_"

    with open(f'{directory}/input_batch_{suffix+str(batch_id)}.pkl', 'rb') as handle:
        input_batch = pickle.load(handle)
    with open(f'{directory}/target_cost_{suffix+str(batch_id)}.pkl', 'rb') as handle:
        target_cost = pickle.load(handle)
    with open(f'{directory}/target_cardinality_{suffix+str(batch_id)}.pkl', 'rb') as handle:
        target_card = pickle.load(handle)

    return input_batch, target_cost, target_card
