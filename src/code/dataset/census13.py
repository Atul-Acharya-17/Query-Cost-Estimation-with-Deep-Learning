import pandas as pd

from ..constants import DATA_ROOT, SAMPLE_NUM
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