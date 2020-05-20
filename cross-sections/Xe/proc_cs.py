import numpy as np
import pandas as pd
import os

files = []
for r, d, f in os.walk('.'):
    for file in f:
        if '.txt' in file:
            files.append(file)

for file in files:
    data = ''
    with open(file) as f:
        data = f.read()
        data = data.replace(' ', '').replace('\t',';')
    with open(file.split('.')[0] + '.csv', 'w') as f:
        f.write(data)