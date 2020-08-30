import csv
import sys

import joblib
import numpy as np
import pandas as pd
import pydotplus
from rdp import rdp
from sklearn import metrics, model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

clf = joblib.load('isDraw/s1_RF.pkl')

file = sys.argv[1]

path = file.rsplit("/",1)[0]

data = pd.read_csv(file)
x = data["X"].values
y = data["Y"].values

def standardize(x):
	return (x - np.nanmean(x))/(np.nanstd(x))

n = 3
dx = [[] for i in range(2*n)]
dy = [[] for i in range(2*n)]
for i in range(len(x)):
    index = 0
    for j in range(-n,0):
        if i+j < 0 :
            dx[index].append(np.NaN)
            dy[index].append(np.NaN)
        else:
            dx[index].append(x[i]-x[i+j])
            dy[index].append(y[i]-y[i+j])
        index += 1 
    for j in range(1,1+n):
        if i+j >= len(x):
            dx[index].append(np.NaN)
            dy[index].append(np.NaN)
        else:
            dx[index].append(x[i]-x[i+j])
            dy[index].append(y[i]-y[i+j])
        index += 1

#fill = np.nanmax(dx) if np.nanmax(dx)>np.nanmax(dy) else np.nanmax(dy)

imp = SimpleImputer(missing_values=np.nan,strategy='constant')
dx = imp.fit_transform(dx)
dy = imp.fit_transform(dy)

dx = pd.DataFrame(np.array([standardize(d) for d in dx]).T,columns=["dx"+str(i) for i in range(0-n,1+n) if i != 0])
dy = pd.DataFrame(np.array([standardize(d) for d in dy]).T,columns=["dy"+str(i) for i in range(0-n,1+n) if i != 0])

features = dx.join(dy)
features = np.array(features)

points = np.array(data)[:,1:3]

mask = rdp(points, algo = 'iter', return_mask = True, epsilon = 8)

prob = clf.predict_proba(features)
y_pre = clf.predict(features)

for i in y_pre:
    if i != 1:
        i = -1

results = pd.DataFrame({"score":y_pre,"turn":mask})

tra = data.join(results)

tra.to_csv(path+"/trajectory.csv",index=False)

rows = []
count = 0
socer = 0
# rows.append({"frame":0,"X":points[0,0],"Y":points[0,1],"score":0})
for i in range(len(points)):
    socer += y_pre[i]
    count += 1
    if mask[i] == True:
        row = {"frame":i,"X":points[i,0],"Y":points[i,1],"score":socer/count}
        rows.append(row)
        socer = 0
        count = 0

with open(path+"/dp.csv","w") as f:
    rst_csv = csv.DictWriter(f,["frame","X","Y","score"])
    rst_csv.writeheader()
    rst_csv.writerows(rows)
