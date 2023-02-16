physic_ops_id = {
    'Materialize':1,
    'Sort':2, 
    'Hash':3, 
    'Merge Join':4, 
    'Bitmap Index Scan':5,
    'Index Only Scan':6, 
    'BitmapAnd':7, 
    'Nested Loop':8, 
    'Aggregate':9, 
    'Result':10,
    'Hash Join':11, 
    'Seq Scan':12, 
    'Bitmap Heap Scan':13, 
    'Index Scan':14, 
    'BitmapOr':15,
    'Gather': 16
}

id2op = {}

for key, value in physic_ops_id.items():
    id2op[value] = key

strategy_id = {
    'Plain':1
}

compare_ops_id = {
    '=':1, 
    '>':2, 
    '<':3, 
    '!=':4, 
    '~~':5, 
    '!~~':6, 
    '!Null': 7, 
    '>=':8, 
    '<=':9
}

bool_ops_id = {
    'AND':1,
    'OR':2
}
