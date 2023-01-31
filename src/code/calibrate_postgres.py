import psycopg2
import json
from .constants import DATABASE_URL


class DBMS():

    def __init__(self):

        self.conn = psycopg2.connect(DATABASE_URL)
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()
    
    def query(self, query):
        self.cursor.execute(query)
        res = self.cursor.fetchall()
        return res
    
    def disable_parallel(self):
        sql = 'SET max_parallel_workers_per_gather = 0;'
        self.cursor.execute(sql)
        
    def query_no_result(self, query):
        self.cursor.execute(query)
        
        
def calibrate():
    
    dbms = DBMS()
    
    load_table = 'SET work_mem=2097151;'
    
    unload_table = 'SET work_mem=4096;'
        
    q_1 = 'SELECT * FROM aka_name;'
    q_2 = 'SELECT COUNT(*) FROM aka_name;'
    
    q_3 = 'SELECT * FROM aka_name where id < 15000;'
    
    q_4 = 'SELECT * FROM aka_name;'
    
    q_5 = 'SELECT * FROM cast_info where movie_id < 16000' # Unclustered index
    
    dbms.query_no_result("SET enable_bitmapscan=False;")
    
    dbms.disable_parallel()
    
    i = 0
    
    values = {
        'c_t': [],
        'c_i': [],
        'c_o': [],
        'c_r': [],
        'c_s': []
    }
    
    while i < 20:
    
        dbms.query_no_result(load_table)

        # dbms.query_no_result(discard)
        
        # Get c_t
        query = f'explain (analyze, format json) {q_1}'
        result = dbms.query(query)
        plan = result[0][0][0]['Plan']
        
        t_1 = plan['Actual Total Time']
        n_t = plan['Actual Rows']    
        c_t = t_1 / n_t
        
        # Get c_o
        # dbms.query_no_result(discard)
        query = f'explain (analyze, format json) {q_2}'
        result = dbms.query(query)
        plan = result[0][0][0]['Plan']
        
        t_2 = plan['Actual Total Time']
        n_o = n_t
        c_o = (t_2 - t_1) / n_o

        # Get c_i
        # dbms.query_no_result(discard)
        query = f'explain (analyze, format json) {q_3}'
        result = dbms.query(query)
        plan = result[0][0][0]['Plan']
        
        t_3 = plan['Actual Total Time']
        n_i = plan['Actual Rows']
        c_i = (t_3 - n_i*c_o - n_i*c_t) / n_i # n_i = n_o = n_t in this case  
        
        # Get c_s
        # dbms.query_no_result(discard)
        dbms.query_no_result(unload_table)
        block_size = f"SELECT current_setting('block_size');"
        block_size = dbms.query(block_size)
        block_size = int(block_size[0][0])
        
        relation_size = f"select pg_relation_size('aka_name');"
        relation_size = dbms.query(relation_size)
        relation_size = int(relation_size[0][0])
        
        n_s = relation_size / block_size
        
        query = f"explain (analyze, format json) {q_4}"
        result = dbms.query(query)
        plan = result[0][0][0]['Plan']
        t_4 = plan['Actual Total Time']
        c_s = (t_4 - t_1) / n_s 
        
        # Get c_ r
        # dbms.query_no_result(discard)
        query = f"explain (analyze, format json) {q_5}"
        result = dbms.query(query)
        plan = result[0][0][0]['Plan']
        t_5 = plan['Actual Total Time']
        n_r = plan['Actual Rows'] # Assume pages = tuples since random access
        
        query = f"SELECT round({n_r}::numeric/reltuples::numeric, 4) FROM pg_class WHERE relname = 'cast_info';"

        selectivity = float(dbms.query(query)[0][0])
        
        query = "select pg_table_size('movie_id_cast_info')"
        index_size = int(dbms.query(query)[0][0])
        
        n_s = index_size / block_size * selectivity

        c_r = (t_5 - n_r*(c_o + c_i + c_t) - n_s * c_s) / n_r

        
        if c_t > 0 and c_o > 0 and c_i > 0 and c_r > 0 and c_s > 0:
            values['c_i'].append(c_i)
            values['c_t'].append(c_t)
            values['c_o'].append(c_o)
            values['c_r'].append(c_r)
            values['c_s'].append(c_s)
            i+=1
            print(f'{i}/20', end='\r')
    
    # print(f't_1 : {t_1}')
    # print(f't_2 : {t_2}')
    # print(f't_3 : {t_3}')
    # print(f't_4 : {t_4}')
    # print(f't_5 : {t_5}')
    
    # calibrated_values = {
    #     'c_t': c_t,
    #     'c_o': c_o,
    #     'c_i': c_i,
    #     'c_r': c_r,
    #     'c_s': c_s
    # }
    
    c_t = sum(values['c_t']) / len(values['c_t'])
    c_o = sum(values['c_o']) / len(values['c_o'])
    c_i = sum(values['c_i']) / len(values['c_i'])
    c_r = sum(values['c_r']) / len(values['c_r'])
    c_s = sum(values['c_s']) / len(values['c_s'])

    print(f'c_s : {c_s}')  
    print(f'c_r : {c_r}')
    print(f'c_i : {c_i}') 
    print(f'c_t : {c_t}') 
    print(f'c_o : {c_o}')
    
    # with open('./calibrated_values.json', 'w') as f:
    #     json.dump(calibrated_values, f)
        
    with open('./data/calibration.sql', 'w') as f:
        f.write(f'SET cpu_tuple_cost={c_t:.15f}\n')
        f.write(f'SET cpu_operator_cost={c_o:.15f}\n')
        f.write(f'SET cpu_index_tuple_cost={c_i:.15f}\n')
        f.write(f'SET random_page_cost={c_r:.15f}\n')
        f.write(f'SET seq_page_cost={c_s:.15f}\n')
    

if __name__ == '__main__':
    calibrate()