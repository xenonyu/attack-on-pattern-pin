import csv
import sys

import numpy as np
import pandas as pd
from rdp import rdp

# Path of csv
file = sys.argv[1]

dir = str(file).rsplit("/",1)[0]

print(dir)
data = pd.read_csv(file)
x = data["X"].values
y = data["Y"].values

points = []

for i in range(len(x)):
    point = [x[i],y[i]]
    points.append(point)

mask = rdp(points, algo = 'iter', return_mask = True, epsilon = 4)

rows = []
for i in range(len(points)):
    if mask[i] == True:
        row = {"frame":i,"X":points[i][0],"Y":points[i][1]}
        rows.append(row)


with open(dir+"/dp.csv","w") as f:
    rst_csv = csv.DictWriter(f,["frame","X","Y"])
    rst_csv.writeheader()
    rst_csv.writerows(rows)
