import pandas as pd

from ..constants import DATA_ROOT, SAMPLE_NUM
from ..plan.utils import encode_sample

data = {}
sample = {}

data["dmv11_original"] = pd.read_csv(f"{DATA_ROOT}/dmv11/original.csv")


columns = {
    col.lower(): idx for idx, col in enumerate(data["dmv11_original"].columns)
}

data["dmv11_original"].columns = columns

sample["dmv11_original"] = data['dmv11_original'].sample(n=min(SAMPLE_NUM, len(data['dmv11_original'])))

tables = ["dmv11_original"]

indexes = []

indexes_id = {}

tables_id = {}

indexes_id = {}

columns_id = {}

for idx, table in enumerate(tables):
    tables_id[table] = idx + 1

    for column, id in columns.items():
        columns_id[f"{table}.{column}"] = id + 1


str_columns = ['record_type', 'registration_class', 'state', 'county', 'body_type', 'fuel_type', 'color', 'scofflaw_indicator', 'suspension_indicator', 'revocation_indicator']


one_hot_encoding = {
    column: {
        unique_values : "0" * list(pd.unique(data['dmv11_original'][column])).index(unique_values) + "1" + "0" * (len(pd.unique(data['dmv11_original'][column])) - list(pd.unique(data['dmv11_original'][column])).index(unique_values) - 1)
            
        for unique_values in pd.unique(data['dmv11_original'][column])
    }

    for column in str_columns
}

num_cols = ['reg_valid_date']

min_max_column = {}

min_max_column['dmv11_original'] = {
    column: {
        'max': max(data['dmv11_original'][column]),
        'min': min(data['dmv11_original'][column])
    }
    for column in num_cols
}

max_string_dim = max([len(pd.unique(data['dmv11_original'][column])) for column in str_columns])


def get_representation(value, column):
    one_hot = one_hot_encoding[column]
    if value not in one_hot:
        return [0 for _ in range(pd.unique(data['dmv11_original'][column]))]
    return encode_sample(one_hot[value])