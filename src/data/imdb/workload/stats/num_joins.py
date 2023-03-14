import pandas as pd
import matplotlib.pyplot as plt

files = ['synthetic.csv', 'job-light.csv']

plt.style.use('seaborn-whitegrid')

joins = []

for file in files:
    num_joins = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for row in lines:
            components = row.split('#')
            
            join_component = components[1]

            join_component = join_component.split(',')
            num = len(join_component)
            num_joins.append(num)

    joins.append(num_joins)

plt.boxplot(joins, labels=['Synthetic500', 'Job-light'])
plt.ylabel('Number of joins')
plt.show()


            