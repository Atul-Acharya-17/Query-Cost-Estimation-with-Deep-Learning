name = 'job-light'

filename = f'{name}.csv'
output_filename = f'{name}.sql'


if __name__ == '__main__':

    with open(output_filename, 'w') as sql_file:
        with open(filename, 'r') as f:
            lines = f.readlines()

            for line in lines:

                tables, joins, predicates, card = line.split("#")

                joins = joins.split(',')
                join_str = ' AND '.join(joins)

                predicates = predicates.split(',')
                predicates = [predicates[3*i] + predicates[3*i+1] + predicates[3*i+2] for i in range(int(len(predicates)/3))]
                predicate_str = ' AND '.join(predicates)

                where_clause = ''
                if len(join_str) > 0 and len(predicate_str) > 0:
                    where_clause = f'{join_str} AND {predicate_str}'
                elif len(join_str) > 0:
                    where_clause = f'{join_str}'
                elif len(predicate_str) > 0:
                    where_clause = f'{predicate_str}'

                sql = f'SELECT * FROM {tables} WHERE {where_clause};'

                sql_file.write(sql + '\n')
