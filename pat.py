import sys

import numpy as np
import pandas as pd

file = sys.argv[1]

f = open(file)

pats = f.readlines()

map = [ (-1,1),(0,1),(1,1),
        (-1,0),(0,0),(1,0),
        (-1,-1),(0,-1),(1,-1)]


for pat in pats:
    pat = pat.split(" --> ")
    name = ''.join(pat)[0:-1]
    temp = []
    x = []
    y = []
    for p in pat:
        x.append(map[int(p)-1][0])
        y.append(map[int(p)-1][1])
    dataframe = pd.DataFrame({'X':x,'Y':y})
    dataframe.to_csv('pats/'+name+'.csv')
