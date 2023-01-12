from ..constants import DATA_ROOT, SAMPLE_NUM
import pandas as pd
from ..plan.utils import encode_sample

data = {}
sample = {}

data["census13_original"] = pd.read_csv(f"{DATA_ROOT}/census13/original.csv")


columns = {
    col: idx for idx, col in enumerate(data["census13_original"].columns)
}

data["census13_original"].columns = columns

sample["census13_original"] = data['census13_original'].sample(n=min(SAMPLE_NUM, len(data['census13_original'])))

tables = ["census13_original"]

indexes = []

indexes_id = {}

tables_id = {}

indexes_id = {}

columns_id = {}

for idx, table in enumerate(tables):
    tables_id[table] = idx + 1

    for column, id in columns.items():
        columns_id[f"{table}.{column}"] = id + 1


str_columns = ['workclass', 'education', 'marital_status', 'occupation', 'race', 'relationship', 'sex', 'native_country']


one_hot_encoding = {
    column: {
        unique_values : "0" * list(pd.unique(data['census13_original'][column])).index(unique_values) + "1" + "0" * (len(pd.unique(data['census13_original'][column])) - list(pd.unique(data['census13_original'][column])).index(unique_values) - 1)
            
        for unique_values in pd.unique(data['census13_original'][column])
    }

    for column in str_columns
}

num_cols = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']

min_max_column = {}

min_max_column['census13_original'] = {
    column: {
        'max': max(data['census13_original'][column]),
        'min': min(data['census13_original'][column])
    }
    for column in num_cols
}

max_string_dim = max([len(pd.unique(data['census13_original'][column])) for column in str_columns])


def get_representation(value, column):
    one_hot = one_hot_encoding[column]
    if value not in one_hot:
        return [0 for _ in range(pd.unique(data['census13_original'][column]))]
    return encode_sample(one_hot[value])

# def initialize_dict():

#     tokens = []
#     for idx, row in data["census13_original"].iterrows():
#         if idx % 100 == 0:
#             print('census13_original ', idx, ' / ', len(data["census13_original"]))

#         sentence = []
#         sentence.append('workclass_' + str(row['workclass']))
#         sentence.append('education_' + str(row['education']))
#         sentence.append('marital_status_' + str(row['marital_status']))
#         sentence.append('occupation_' + str(row['occupation']))
#         sentence.append('relationship_' + str(row['relationship']))
#         sentence.append('sex_' + str(row['sex']))
#         sentence.append('native_country_' + str(row['native_country']))

#         tokens.append(sentence)

#     return tokens


# def train_word2vec():
#     cores = multiprocessing.cpu_count()
#     w2v_model = Word2Vec(min_count=5,
#                         window=5,
#                         vector_size=500,
#                         alpha=0.03,
#                         min_alpha=0.0007,
#                         negative=20,
#                         workers=cores - 10)

#     sentences = initialize_dict()

#     t = time()
#     w2v_model.build_vocab(sentences, progress_per=10000)
#     print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

#     t = time()
#     w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
#     print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

#     w2v_model.save("word2vec.model")
#     print('model saved')

#     path = get_tmpfile("wordvectors.kv")
#     w2v_model.wv.save(path)
#     print('word saved')

#     print(w2v_model.wv)

# if __name__ == '__main__':
#     train_word2vec()