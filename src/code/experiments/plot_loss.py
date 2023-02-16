import json
import pandas as pd
import matplotlib.pyplot as plt
from ..constants import RESULT_ROOT


plt.style.use('seaborn-whitegrid')


tree_nn = str(RESULT_ROOT) + '/output/imdb/training_statistics_tree_nn_train.json'
tree_lstm = str(RESULT_ROOT) + '/output/imdb/training_statistics_tree_lstm_train.json'
tree_attn = str(RESULT_ROOT) + '/output/imdb/training_statistics_tree_attn_train.json'
tree_gru = str(RESULT_ROOT) + '/output/imdb/training_statistics_tree_gru_train.json'


files = {
    'TreeNN':tree_nn,
    'TreeGRU': tree_gru,
    'TreeLSTM': tree_lstm,
    'TreeAttn': tree_attn,
}


if __name__ == '__main__':

    cost_losses = {}
    card_losses = {}
    
    cost_losses['Epochs'] = [i for i in range(1, 31)]
    card_losses['Epochs'] = [i for i in range(1, 31)]

    
    for name, file in files.items():
        with open(file, 'r') as f:
            data = json.load(f)

            cost_losses[name] = data['cost_loss_val']
            card_losses[name] = data['card_loss_val']

    cost_df = pd.DataFrame(cost_losses)
    
    card_df = pd.DataFrame(card_losses)
    
    # fig = plt.line(cost_df, x='Epochs', y=list(files.keys()))
    # fig.show()
    
    # fig = plt.line(card_df, x='Epochs', y=list(files.keys()))
    # fig.show()
    
    for name in files:
        plt.plot(cost_losses['Epochs'], cost_losses[name], label=name, marker='o')

    plt.legend(list(files.keys()))
    plt.ylabel("Cost Loss")
    plt.xlabel("Epochs")
    plt.show()

    for name in files:
        plt.plot(card_losses['Epochs'], card_losses[name], label=name, marker='o')

    plt.legend(list(files.keys()))
    plt.ylabel("Cardinality Loss")
    plt.xlabel("Epochs")
    plt.show()
    