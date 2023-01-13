import pandas as pd

from ..constants import DATA_ROOT, SAMPLE_NUM
from ..plan.utils import encode_sample

data = {}
sample = {}

data["forest10_original"] = pd.read_csv(f"{DATA_ROOT}/forest10/original.csv")


columns = {
    col.lower(): idx for idx, col in enumerate(data["forest10_original"].columns)
}

data["forest10_original"].columns = columns

sample["forest10_original"] = data['forest10_original'].sample(n=min(SAMPLE_NUM, len(data['forest10_original'])))

tables = ["forest10_original"]

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
        unique_values : "0" * list(pd.unique(data['forest10_original'][column])).index(unique_values) + "1" + "0" * (len(pd.unique(data['forest10_original'][column])) - list(pd.unique(data['forest10_original'][column])).index(unique_values) - 1)
            
        for unique_values in pd.unique(data['forest10_original'][column])
    }

    for column in str_columns
}

num_cols = ['elevation', 'aspect', 'slope', 'horizontal_distance_to_hydrology', 'vertical_distance_to_hydrology', 'hillshade_9am', 'horizontal_distance_to_roadways', 'hillshade_9am', 'hillshade_noon', 'hillshade_3pm', 'horizontal_distance_to_fire_points']

min_max_column = {}

min_max_column['forest10_original'] = {
    column: {
        'max': max(data['forest10_original'][column]),
        'min': min(data['forest10_original'][column])
    }
    for column in num_cols
}

max_string_dim = 1

# Not used
def get_representation(value, column):
    one_hot = one_hot_encoding[column]
    if value not in one_hot:
        return [0 for _ in range(pd.unique(data['forest10_original'][column]))]
    return encode_sample(one_hot[value])