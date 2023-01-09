from .constants import DATA_ROOT
import pandas as pd


data = {}
sample = {}
sample_num = 1000

data["census13_original"] = pd.read_csv(f"{DATA_ROOT}/census13/original.csv")


census_column = {
    col: idx for idx, col in enumerate(data["census13_original"].columns)
}

data["census13_original"].columns = census_column

sample["census13_original"] = data['census13_original'].sample(n=min(sample_num, len(data['census13_original'])))