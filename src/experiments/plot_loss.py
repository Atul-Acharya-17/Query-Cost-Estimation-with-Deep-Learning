import json


tree_nn = 'imdb/training_statistics_tree_nn_train.json'
tree_lstm = 'imdb/training_statistics_tree_lstm_train.json'
tree_attn = 'imdb/training_statistics_tree_attn_train.json'


files = {
    'TreeNN':tree_nn,
    'TreeLSTM': tree_lstm,
    'TreeAttn': tree_attn
}


if __name__ == '__main__':

    cost_losses = {}
    card_losses = {}
    
    for name, file in enumerate(files):
        with open(file, 'r') as f:
            data = json.load(f)

            cost_losses[name] = data['cost_loss_val']
            card_losses[name] = data['card_loss_val']

    print(cost_losses)
    print(card_losses)



