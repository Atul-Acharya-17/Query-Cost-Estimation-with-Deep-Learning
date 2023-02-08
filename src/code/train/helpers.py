import pickle

def get_batch_job_tree(batch_id, phase, directory, get_unnorm=False):
    suffix = phase + "_"

    with open(f'{directory}/input_batch_{suffix+str(batch_id)}.pkl', 'rb') as handle:
        input_batch = pickle.load(handle)
    with open(f'{directory}/target_cost_{suffix+str(batch_id)}.pkl', 'rb') as handle:
        target_cost = pickle.load(handle)
    with open(f'{directory}/target_cardinality_{suffix+str(batch_id)}.pkl', 'rb') as handle:
        target_card = pickle.load(handle)

    with open(f'{directory}/true_cardinality_{suffix+str(batch_id)}.pkl', 'rb') as handle:
        unnorm_card = pickle.load(handle)

    with open(f'{directory}/true_cost_{suffix+str(batch_id)}.pkl', 'rb') as handle:
        unnorm_cost = pickle.load(handle)

    if get_unnorm:
        return input_batch, target_cost, target_card, unnorm_cost, unnorm_card
        
    return input_batch, target_cost, target_card
