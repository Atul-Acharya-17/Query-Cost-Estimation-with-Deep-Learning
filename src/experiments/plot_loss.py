import json
import pandas as pd
import plotly.express as px
from ..code.constants import RESULT_ROOT


tree_nn = str(RESULT_ROOT) + 'output/imdb/training_statistics_tree_nn_train.json'
tree_lstm = str(RESULT_ROOT) + 'output/imdb/training_statistics_tree_lstm_train.json'
tree_attn = str(RESULT_ROOT) + 'output/imdb/training_statistics_tree_attn_train.json'


files = {
    'TreeNN':tree_nn,
    'TreeLSTM': tree_lstm,
    'TreeAttn': tree_attn
}


if __name__ == '__main__':

    cost_losses = {}
    card_losses = {}
    
    cost_losses['Epochs'] = [i for i in range(1, 31)]
    card_losses['Epochs'] = [i for i in range(1, 31)]

    
    for name, file in enumerate(files):
        with open(file, 'r') as f:
            data = json.load(f)

            cost_losses[name] = data['cost_loss_val']
            card_losses[name] = data['card_loss_val']

    cost_df = pd.DataFrame(cost_losses)
    
    card_df = pd.DataFrame(card_losses)
    
    fig = px.line(cost_df, x='Epochs', y=files.keys())
    fig.show()
    
    fig = px.line(card_df, x='Epochs', y=files.keys())
    fig.show()
    
    