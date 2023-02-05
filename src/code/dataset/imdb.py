import numpy as np
import pandas as pd
import random

from ..constants import DATA_ROOT, SAMPLE_NUM
from ..plan.utils import encode_sample



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


# tables = [key for key in data.keys()]


min_max_column = {}

required = {
    "title": ["id", "kind_id", "production_year"],
    "movie_companies": ["id", "company_id", "movie_id", "company_type"],
    "cast_info": ["id", "movie_id", "person_id", "role_id"],
    "movie_info": ["id", "movie_id", "info_type"],
    "movie_info_idx": ["id", "movie_id", "info_type"],
    "movie_keyword": ["id", "movie_id", "keyword_id"]
}

# for table in required.keys():
#     min_max_column[table] = {
#         column: {
#             'max': max(data[table][column]),
#             'min': min(data[table][column])
#         }
#         for column in required[table]
#     }

# print(min_max_column)

min_max_column = {
    "title": {
        "id": {
            'max': 2528312,
            'min': 1
        },
        "kind_id": {
            'max': 7,
            'min': 1, 
        },
        "production_year": {
            'max': 2019,
            'min': 1880
        }
    },

    "movie_companies": {
        "id": {
            'max': 2609129,
            'min': 1
        },
        "company_id": {
            'max': 234997,
            'min': 1
        },
        "movie_id": {
            'max': 2525745,
            'min': 2
        },
        "company_type_id": {
            'max': 2,
            'min': 1
        }
    },

    "cast_info": {
        "id": {
            'max': 36244344,
            'min': 1
        },
        "movie_id": {
            'max': 2525975,
            'min': 1
        },
        "person_id": {
            'max': 4061926,
            'min': 1
        },
        "role_id": {
            'max': 11,
            'min': 1
        }
    },

    "movie_info": {
        "id": {
            'max': 14835720,
            'min': 1
        },
        "movie_id": {
            'max': 2526430,
            'min': 1
        },
        "info_type_id": {
            'max': 110,
            'min': 1
        }
    
    },

    "movie_info_idx": {
        "id": {
            'max': 1380035,
            'min': 1
        },
        "movie_id": {
            'max': 2525793,
            'min': 2
        },
        "info_type_id": {
            'max': 113,
            'min': 99
        }
    },

    "movie_keyword": {
        "id": {
            'max': 4523930,
            'min': 1
        },
        "movie_id": {
            'max': 2525971,
            'min': 2
        },
        "keyword_id": {
            'max': 134170,
            'min': 1
        }
    }

}


def get_data_and_samples():

    data = {}
    sample = {}

    # Add all tables
    data["aka_name"] = pd.read_csv(str(DATA_ROOT)+'/imdb/aka_name.csv',header=None, delimiter='\n', sep=',', names=aka_name_column.keys())
    print("Read aka_name")

    data["aka_title"] = pd.read_csv(str(DATA_ROOT)+'/imdb/aka_title.csv',header=None, delimiter='\n', sep=',', names=aka_title_column.keys(), dtype={"episode_of_id":pd.Int64Dtype()})
    print("Read aka_title")

    data["cast_info"] = pd.read_csv(str(DATA_ROOT)+'/imdb/cast_info.csv', skiprows=sorted(random.sample(range(36244344),36244344-100000)), header=None, delimiter='\n', sep=',', names=cast_info_column.keys(), dtype={"note":"string", "person_role_id":pd.Int64Dtype(), "nr_order":pd.Int64Dtype()})
    print("Read cast_info")

    data["char_name"] = pd.read_csv(str(DATA_ROOT)+'/imdb/char_name.csv',header=None, delimiter='\n', sep=',', names=char_name_column.keys())
    print("Read char_name")

    data["company_name"] = pd.read_csv(str(DATA_ROOT)+'/imdb/company_name.csv',header=None, delimiter='\n', sep=',', names=company_name_column.keys())
    print("Read company_name")

    data["company_type"] = pd.read_csv(str(DATA_ROOT)+'/imdb/company_type.csv',header=None, delimiter='\n', sep=',', names=company_type_column.keys())
    print("Read company_type")

    data["comp_cast_type"] = pd.read_csv(str(DATA_ROOT)+'/imdb/comp_cast_type.csv',header=None, delimiter='\n', sep=',', names=comp_cast_type_column.keys())
    print("Read comp_cast_type")

    data["complete_cast"] = pd.read_csv(str(DATA_ROOT)+'/imdb/complete_cast.csv',header=None, delimiter='\n', sep=',', names=complete_cast_column.keys())
    print("Read complete_cast")

    data["info_type"] = pd.read_csv(str(DATA_ROOT)+'/imdb/info_type.csv',header=None, delimiter='\n', sep=',', names=info_type_column.keys())
    print("Read info_type")

    data["keyword"] = pd.read_csv(str(DATA_ROOT)+'/imdb/keyword.csv',header=None, delimiter='\n', sep=',', names=keyword_column.keys())
    print("Read keyword")

    data["kind_type"] = pd.read_csv(str(DATA_ROOT)+'/imdb/kind_type.csv',header=None, delimiter='\n', sep=',', names=kind_type_column.keys())
    print("Read kind_type")

    data["link_type"] = pd.read_csv(str(DATA_ROOT)+'/imdb/link_type.csv',header=None, delimiter='\n', sep=',', names=link_type_column.keys())
    print("Read link_type")

    data["movie_companies"] = pd.read_csv(str(DATA_ROOT)+'/imdb/movie_companies.csv',header=None, delimiter='\n', sep=',', names=movie_companies_column.keys())
    print("Read movie_companies")

    data["movie_info"] = pd.read_csv(str(DATA_ROOT)+'/imdb/movie_info.csv',header=None, delimiter='\n', sep=',', names=movie_info_column.keys())
    print("Read movie_info")

    data["movie_info_idx"] = pd.read_csv(str(DATA_ROOT)+'/imdb/movie_info_idx.csv',header=None, delimiter='\n', sep=',', names=movie_info_idx_column.keys())
    print("Read movie_info_idx")

    data["movie_keyword"] = pd.read_csv(str(DATA_ROOT)+'/imdb/movie_keyword.csv',header=None, delimiter='\n', sep=',', names=movie_keyword_column.keys())
    print("Read movie_keyword")

    data["movie_link"] = pd.read_csv(str(DATA_ROOT)+'/imdb/movie_link.csv',header=None, delimiter='\n', sep=',', names=movie_link_column.keys())
    print("Read movie_link")

    data["name"] = pd.read_csv(str(DATA_ROOT)+'/imdb/name.csv',header=None, delimiter='\n', sep=',', names=name_column.keys())
    print("Read name")

    data["person_info"] = pd.read_csv(str(DATA_ROOT)+'/imdb/person_info.csv',header=None, delimiter='\n', sep=',', names=person_info_column.keys())
    print("Read person_info")

    data["role_type"] = pd.read_csv(str(DATA_ROOT)+'/imdb/role_type.csv',header=None, delimiter='\n', sep=',', names=role_type_column.keys())
    print("Read role_type")

    data["title"] = pd.read_csv(str(DATA_ROOT)+'/imdb/title.csv',header=None, delimiter='\n', sep=',', names=title_column.keys())
    print("Read title")

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

    return data, sample


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

tables_id = {'aka_name': 1, 'aka_title': 2, 'cast_info': 3, 'char_name': 4, 'company_name': 5, 'company_type': 6, 'comp_cast_type': 7, 'complete_cast': 8, 'info_type': 9, 'keyword': 10, 'kind_type': 11, 'link_type': 12, 'movie_companies': 13, 'movie_info': 14, 'movie_info_idx': 15, 'movie_keyword': 16, 'movie_link': 17, 'name': 18, 'person_info': 19, 'role_type': 20, 'title': 21}

# data, samples = get_data_and_samples()

# for idx, table in enumerate(data.keys()):
#     tables_id[table] = idx + 1

# print(tables_id)

columns_id = {}

# id = 1
# for idx, table in enumerate(tables):
#     tables_id[table] = idx + 1

#     for column in data[table].columns:
#         columns_id[f"{table}.{column}"] = id
#         id += 1

columns_id = {'aka_name.id': 1, 'aka_name.person_id': 2, 'aka_name.name': 3, 'aka_name.imdb_index': 4, 'aka_name.name_pcode_cf': 5, 'aka_name.name_pcode_nf': 6, 'aka_name.surname_pcode': 7, 'aka_name.md5sum': 8, 'aka_title.id': 9, 'aka_title.movie_id': 10, 'aka_title.title': 11, 'aka_title.imdb_index': 12, 'aka_title.kind_id': 13, 'aka_title.production_year': 14, 'aka_title.phonetic_code': 15, 'aka_title.episode_of_id': 16, 'aka_title.season_nr': 17, 'aka_title.episode_nr': 18, 'aka_title.note': 19, 'aka_title.md5sum': 20, 'cast_info.id': 21, 'cast_info.person_id': 22, 'cast_info.movie_id': 23, 'cast_info.person_role_id': 24, 'cast_info.note': 25, 'cast_info.nr_order': 26, 'cast_info.role_id': 27, 'char_name.id': 28, 'char_name.name': 29, 'char_name.imdb_index': 30, 'char_name.imdb_id': 31, 'char_name.name_pcode_nf': 32, 'char_name.surname_pcode': 33, 'char_name.md5sum': 34, 'company_name.id': 35, 'company_name.name': 36, 'company_name.country_code': 37, 'company_name.imdb_id': 38, 'company_name.name_pcode_nf': 39, 'company_name.name_pcode_sf': 40, 'company_name.md5sum': 41, 'company_type.id': 42, 'company_type.kind': 43, 'comp_cast_type.id': 44, 'comp_cast_type.kind': 45, 'complete_cast.id': 46, 'complete_cast.movie_id': 47, 'complete_cast.subject_id': 48, 'complete_cast.status_id': 49, 'info_type.id': 50, 'info_type.info': 51, 'keyword.id': 52, 'keyword.keyword': 53, 'keyword.phonetic_code': 54, 'kind_type.id': 55, 'kind_type.kind': 56, 'link_type.id': 57, 'link_type.link': 58, 'movie_companies.id': 59, 'movie_companies.movie_id': 60, 'movie_companies.company_id': 61, 'movie_companies.company_type_id': 62, 'movie_companies.note': 63, 'movie_info.id': 64, 'movie_info.movie_id': 65, 'movie_info.info_type_id': 66, 'movie_info.info': 67, 'movie_info.note': 68, 'movie_info_idx.id': 69, 'movie_info_idx.movie_id': 70, 'movie_info_idx.info_type_id': 71, 'movie_info_idx.info': 72, 'movie_info_idx.note': 73, 'movie_keyword.id': 74, 'movie_keyword.movie_id': 75, 'movie_keyword.keyword_id': 76, 'movie_link.id': 77, 'movie_link.movie_id': 78, 'movie_link.linked_movie_id': 79, 'movie_link.link_type_id': 80, 'name.id': 81, 'name.name': 82, 'name.imdb_index': 83, 'name.imdb_id': 84, 'name.gender': 85, 'name.name_pcode_cf': 86, 'name.name_pcode_nf': 87, 'name.surname_pcode': 88, 'name.md5sum': 89, 'person_info.id': 90, 'person_info.person_id': 91, 'person_info.info_type_id': 92, 'person_info.info': 93, 'person_info.note': 94, 'role_type.id': 95, 'role_type.role': 96, 'title.id': 97, 'title.title': 98, 'title.imdb_index': 99, 'title.kind_id': 100, 'title.production_year': 101, 'title.imdb_id': 102, 'title.phonetic_code': 103, 'title.episode_of_id': 104, 'title.season_nr': 105, 'title.episode_nr': 106, 'title.series_years': 107, 'title.md5sum': 108}

# No string values in imdb workload so not implemented for now
one_hot_encoding = {}
str_columns = []
max_string_dim = 1

def get_representation(value, column):
    raise NotImplementedError