import pandas as pd

from ..constants import DATA_ROOT, SAMPLE_NUM
from ..plan.utils import encode_sample

data = {}
sample = {}

data["power7_original"] = pd.read_csv(f"{DATA_ROOT}/power7/original.csv")


columns = {
    col: idx for idx, col in enumerate(data["power7_original"].columns)
}

data["power7_original"].columns = columns

sample["power7_original"] = data['power7_original'].sample(n=min(SAMPLE_NUM, len(data['power7_original'])))

tables = ["power7_original"]

indexes = []

indexes_id = {}

tables_id = {}

indexes_id = {}

columns_id = {}

for idx, table in enumerate(tables):
    tables_id[table] = idx + 1

    for column, id in columns.items():
        columns_id[f"{table}.{column}"] = id + 1


str_columns = []


one_hot_encoding = {
    column: {
        unique_values : "0" * list(pd.unique(data['census13_original'][column])).index(unique_values) + "1" + "0" * (len(pd.unique(data['census13_original'][column])) - list(pd.unique(data['census13_original'][column])).index(unique_values) - 1)
            
        for unique_values in pd.unique(data['census13_original'][column])
    }

    for column in str_columns
}

num_cols = [ 'global_active_power', 'global_reactive_power', 'voltage', 'global_intensity', 'sub_metering_1', 'sub_metering_2', 'sub_metering_3']

min_max_column = {}

min_max_column['power7_original'] = {
    column: {
        'max': max(data['power7_original'][column]),
        'min': min(data['power7_original'][column])
    }
    for column in num_cols
}

max_string_dim = 1

# Not used
def get_representation(value, column):
    one_hot = one_hot_encoding[column]
    if value not in one_hot:
        return [0 for _ in range(pd.unique(data['power7_original'][column]))]
    return encode_sample(one_hot[value])