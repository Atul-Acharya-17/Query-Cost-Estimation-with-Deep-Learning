import numpy as np
import pandas as pd

from ..constants import DATA_ROOT, SAMPLE_NUM
from ..plan.utils import encode_sample


data = {}
sample = {}

# Add all tables
data["aka_name"] = pd.read_csv(DATA_ROOT+'/aka_name.csv',header=None)
data["aka_title"] = pd.read_csv(DATA_ROOT+'/aka_title.csv',header=None)
data["cast_info"] = pd.read_csv(DATA_ROOT+'/cast_info.csv',header=None)
data["char_name"] = pd.read_csv(DATA_ROOT+'/char_name.csv',header=None)
data["company_name"] = pd.read_csv(DATA_ROOT+'/company_name.csv',header=None)
data["company_type"] = pd.read_csv(DATA_ROOT+'/company_type.csv',header=None)
data["comp_cast_type"] = pd.read_csv(DATA_ROOT+'/comp_cast_type.csv',header=None)
data["complete_cast"] = pd.read_csv(DATA_ROOT+'/complete_cast.csv',header=None)
data["info_type"] = pd.read_csv(DATA_ROOT+'/info_type.csv',header=None)
data["keyword"] = pd.read_csv(DATA_ROOT+'/keyword.csv',header=None)
data["kind_type"] = pd.read_csv(DATA_ROOT+'/kind_type.csv',header=None)
data["link_type"] = pd.read_csv(DATA_ROOT+'/link_type.csv',header=None)
data["movie_companies"] = pd.read_csv(DATA_ROOT+'/movie_companies.csv',header=None)
data["movie_info"] = pd.read_csv(DATA_ROOT+'/movie_info.csv',header=None)
data["movie_info_idx"] = pd.read_csv(DATA_ROOT+'/movie_info_idx.csv',header=None)
data["movie_keyword"] = pd.read_csv(DATA_ROOT+'/movie_keyword.csv',header=None)
data["movie_link"] = pd.read_csv(DATA_ROOT+'/movie_link.csv',header=None)
data["name"] = pd.read_csv(DATA_ROOT+'/name.csv',header=None)
data["person_info"] = pd.read_csv(DATA_ROOT+'/person_info.csv',header=None)
data["role_type"] = pd.read_csv(DATA_ROOT+'/role_type.csv',header=None)
data["title"] = pd.read_csv(DATA_ROOT+'/title.csv',header=None)

aka_name_column = {
    'id':0,
    'person_id':1,
    'name':2,
    'imdb_index':3,
    'name_pcode_cf':4,
    'name_pcode_nf':5,
    'surname_pcode':6,
    'md5sum':7
}

aka_title_column = {
    'id':0,
    'movie_id':1,
    'title':2,
    'imdb_index':3,
    'kind_id':4,
    'production_year':5,
    'phonetic_code':6,
    'episode_of_id':7,
    'season_nr':8,
    'episode_nr':9,
    'note':10,
    'md5sum':11
}

cast_info_column = {
    'id':0,
    'person_id':1,
    'movie_id':2,
    'person_role_id':3,
    'note':4,
    'nr_order':5,
    'role_id':6
}

char_name_column = {
    'id':0,
    'name':1,
    'imdb_index':2,
    'imdb_id':3,
    'name_pcode_nf':4,
    'surname_pcode':5,
    'md5sum':6
}

comp_cast_type_column = {
    'id':0,
    'kind':1
}

company_name_column = {
    'id':0,
    'name':1,
    'country_code':2,
    'imdb_id':3,
    'name_pcode_nf':4,
    'name_pcode_sf':5,
    'md5sum':6
}

company_type_column = {
    'id':0,
    'kind':1
}

complete_cast_column = {
    'id':0,
    'movie_id':1,
    'subject_id':2,
    'status_id':3
}

info_type_column = {
    'id':0,
    'info':1
}

keyword_column = {
    'id':0,
    'keyword':1,
    'phonetic_code':2
}

kind_type_column = {
    'id':0,
    'kind':1
}

link_type_column = {
    'id':0,
    'link':1
}

movie_companies_column = {
    'id':0,
    'movie_id':1,
    'company_id':2,
    'company_type_id':3,
    'note':4
}

movie_info_idx_column = {
    'id':0,
    'movie_id':1,
    'info_type_id':2,
    'info':3,
    'note':4
}

movie_keyword_column = {
    'id':0,
    'movie_id':1,
    'keyword_id':2
}

movie_link_column = {
    'id':0,
    'movie_id':1,
    'linked_movie_id':2,
    'link_type_id':3
}


name_column = {
    'id':0,
    'name':1,
    'imdb_index':2,
    'imdb_id':3,
    'gender':4,
    'name_pcode_cf':5,
    'name_pcode_nf':6,
    'surname_pcode':7,
    'md5sum':8
}

role_type_column = {
    'id':0,
    'role':1
}

title_column = {
    'id':0,
    'title':1,
    'imdb_index':2,
    'kind_id':3,
    'production_year':4,
    'imdb_id':5,
    'phonetic_code':6,
    'episode_of_id':7,
    'season_nr':8,
    'episode_nr':9,
    'series_years':10,
    'md5sum':11
}

movie_info_column = {
    'id':0,
    'movie_id':1,
    'info_type_id':2,
    'info':3,
    'note':4
}

person_info_column = {
    'id':0,
    'person_id':1,
    'info_type_id':2,
    'info':3,
    'note':4
}


data["aka_name"].columns = aka_name_column
data["aka_title"].columns = aka_title_column
data["cast_info"].columns = cast_info_column
data["char_name"].columns = char_name_column
data["company_name"].columns = company_name_column
data["company_type"].columns = company_type_column
data["comp_cast_type"].columns = comp_cast_type_column
data["complete_cast"].columns = complete_cast_column
data["info_type"].columns = info_type_column
data["keyword"].columns = keyword_column
data["kind_type"].columns = kind_type_column
data["link_type"].columns = link_type_column
data["movie_companies"].columns = movie_companies_column
data["movie_info"].columns = movie_info_column
data["movie_info_idx"].columns = movie_info_idx_column
data["movie_keyword"].columns = movie_keyword_column
data["movie_link"].columns = movie_link_column
data["name"].columns = name_column
data["person_info"].columns = person_info_column
data["role_type"].columns = role_type_column
data["title"].columns = title_column

sample['aka_name'] = data['aka_name'].sample(n=min(SAMPLE_NUM,len(data['aka_name'])))
sample['aka_title'] = data['aka_title'].sample(n=min(SAMPLE_NUM,len(data['aka_title'])))
sample['cast_info'] = data['cast_info'].sample(n=min(SAMPLE_NUM,len(data['cast_info'])))
sample['char_name'] = data['char_name'].sample(n=min(SAMPLE_NUM,len(data['char_name'])))
sample['company_name'] = data['company_name'].sample(n=min(SAMPLE_NUM,len(data['company_name'])))
sample['company_type'] = data['company_type'].sample(n=min(SAMPLE_NUM,len(data['company_type'])))
sample['comp_cast_type'] = data['comp_cast_type'].sample(n=min(SAMPLE_NUM,len(data['comp_cast_type'])))
sample['complete_cast'] = data['complete_cast'].sample(n=min(SAMPLE_NUM,len(data['complete_cast'])))
sample['info_type'] = data['info_type'].sample(n=min(SAMPLE_NUM,len(data['info_type'])))
sample['keyword'] = data['keyword'].sample(n=min(SAMPLE_NUM,len(data['keyword'])))
sample['kind_type'] = data['kind_type'].sample(n=min(SAMPLE_NUM,len(data['kind_type'])))
sample['link_type'] = data['link_type'].sample(n=min(SAMPLE_NUM,len(data['link_type'])))
sample['movie_companies'] = data['movie_companies'].sample(n=min(SAMPLE_NUM,len(data['movie_companies'])))
sample['movie_info'] = data['movie_info'].sample(n=min(SAMPLE_NUM,len(data['movie_info'])))
sample['movie_info_idx'] = data['movie_info_idx'].sample(n=min(SAMPLE_NUM,len(data['movie_info_idx'])))
sample['movie_keyword'] = data['movie_keyword'].sample(n=min(SAMPLE_NUM,len(data['movie_keyword'])))
sample['movie_link'] = data['movie_link'].sample(n=min(SAMPLE_NUM,len(data['movie_link'])))
sample['name'] = data['name'].sample(n=min(SAMPLE_NUM,len(data['name'])))
sample['person_info'] = data['person_info'].sample(n=min(SAMPLE_NUM,len(data['person_info'])))
sample['role_type'] = data['role_type'].sample(n=min(SAMPLE_NUM,len(data['role_type'])))
sample['title'] = data['title'].sample(n=min(SAMPLE_NUM,len(data['title'])))

tables = [key for key in data.keys()]

indexes = ['aka_name_pkey', 'aka_title_pkey', 'cast_info_pkey', 'char_name_pkey',
               'comp_cast_type_pkey', 'company_name_pkey', 'company_type_pkey', 'complete_cast_pkey',
               'info_type_pkey', 'keyword_pkey', 'kind_type_pkey', 'link_type_pkey', 'movie_companies_pkey',
               'movie_info_idx_pkey', 'movie_keyword_pkey', 'movie_link_pkey', 'name_pkey', 'role_type_pkey',
               'title_pkey', 'movie_info_pkey', 'person_info_pkey', 'company_id_movie_companies',
               'company_type_id_movie_companies', 'info_type_id_movie_info_idx', 'info_type_id_movie_info',
               'info_type_id_person_info', 'keyword_id_movie_keyword', 'kind_id_aka_title', 'kind_id_title',
               'linked_movie_id_movie_link', 'link_type_id_movie_link', 'movie_id_aka_title', 'movie_id_cast_info',
               'movie_id_complete_cast', 'movie_id_movie_ companies', 'movie_id_movie_info_idx',
               'movie_id_movie_keyword', 'movie_id_movie_link', 'movie_id_movie_info', 'person_id_aka_name',
               'person_id_cast_info', 'person_id_person_info', 'person_role_id_cast_info', 'role_id_cast_info']

indexes_id = {
    index: idx + 1
    for idx, index in enumerate(indexes)
}

tables_id = {}

columns_id = {}

for idx, table in enumerate(tables):
    tables_id[table] = idx + 1

    for column, id in data[table].items():
        columns_id[f"{table}.{column}"] = id + 1
        
print(columns_id)


# No string values in imdb workload so not implemented for now
one_hot_encoding = {}
str_columns = []
max_string_dim = 0

def get_representation(value, column):
    raise NotImplementedError

min_max_column = {}

for table in tables:
    min_max_column[table] = {
        column: {
            'max': max(data[table][column]),
            'min': min(data[table][column])
        }
        for column in data[table].select_dtypes(include=np.number)
    }

# Get min max for numeric cols and get min max

# str_columns = ['record_type', 'registration_class', 'state', 'county', 'body_type', 'fuel_type', 'color', 'scofflaw_indicator', 'suspension_indicator', 'revocation_indicator']


# # one_hot_encoding = {
# #     column: {
# #         unique_values : "0" * list(pd.unique(data['dmv11_original'][column])).index(unique_values) + "1" + "0" * (len(pd.unique(data['dmv11_original'][column])) - list(pd.unique(data['dmv11_original'][column])).index(unique_values) - 1)
            
# #         for unique_values in pd.unique(data['dmv11_original'][column])
# #     }

# #     for column in str_columns
# # }

# numeric_cols = ['reg_valid_date']

# min_max_column = {}

# min_max_column['dmv11_original'] = {
#     column: {
#         'max': max(data['dmv11_original'][column]),
#         'min': min(data['dmv11_original'][column])
#     }
#     for column in numeric_cols
# }

# max_string_dim = max([len(pd.unique(data['dmv11_original'][column])) for column in str_columns])


# def get_representation(value, column):
#     one_hot = one_hot_encoding[column]
#     if value not in one_hot:
#         return [0 for _ in range(pd.unique(data['dmv11_original'][column]))]
#     return encode_sample(one_hot[value])