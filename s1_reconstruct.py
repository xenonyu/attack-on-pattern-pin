import sys
from itertools import permutations

import numpy as np
import pandas as pd

file = sys.argv[1]

name = str(file).rsplit("/",1)[0]

data = pd.read_csv(file)
x = data["X"].values
y = data["Y"].values
score = data["score"].values

if len(x) < 3:
    print("Wrong Trajectory!")
    sys.exit()

mid = int(len(x)/2)
ub = []
uf = []
for i in range(mid,0,-1):
    ub.append([(x[i]-x[i+1],y[i]-y[i+1]),(x[i-1]-x[i],y[i-1]-y[i]),score[i]])
for i in range(mid,len(x)-1):
    uf.append([(x[i]-x[i-1],y[i]-y[i-1]),(x[i+1]-x[i],y[i+1]-y[i]),score[i+1]])

def len_v(vector1,vector2):
    return (np.linalg.norm(vector2),np.linalg.norm(vector1))

def cos(vector1,vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    return np.dot(vector1,vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))

def rate(a,b):
    sc1 = cos(a[0],b[0])
    sc2 = cos(a[1],b[1])
    sc3 = cos(a[3],b[3])
    return sc1,sc2,sc3

def conf(a,b):
    theta = 0.9
    thre = 0.7
    if b>thre:
        return ((a[0]+a[1])*theta/2 +(1-theta)*a[2])
    else:
        return ((a[0]+a[1])*theta/2 +(1-theta)*a[2])*(-1)

p3 = list(permutations([0,1,2,3,4,5,6,7,8],3))

m = [(-1,1),(0,1),(1,1),
    (-1,0),(0,0),(1,0),
    (-1,-1),(0,-1),(1,-1)]


vectors = []
for p in p3:
    vectors.append([(m[p[1]][0]-m[p[0]][0],m[p[1]][1]-m[p[0]][1]),(m[p[2]][0]-m[p[1]][0],m[p[2]][1]-m[p[1]][1]),tuple([a+1 for a in p])])


for i in range(len(ub)):
    ub[i].append(len_v(ub[i][0],ub[i][1]))

for i in range(len(uf)):
    uf[i].append(len_v(uf[i][0],uf[i][1]))

for i in range(len(vectors)):
    vectors[i].append(len_v(vectors[i][0],vectors[i][1]))
    

Threshold = 0.6

rb = []
for i in range(len(ub)):
    temp = []
    for v in vectors:
        ra = rate(ub[i],v)
        if (np.array(ra)>Threshold).all():
            temp.append([conf(ra,ub[i][2]),v[2]])
    temp.sort(reverse = True)
    rb.append(temp)
rf = []
for i in range(len(uf)):
    temp = []
    for v in vectors:
        ra = rate(uf[i],v)
        if (np.array(ra)>Threshold).all():
            temp.append([conf(ra,uf[i][2]),v[2]])
    temp.sort(reverse = True)
    rf.append(temp)

res = []
start = (rb.pop(0),rf.pop(0))
for b in start[0]:
    for f in start[1]:
        if f[1] == tuple(reversed(b[1])):
            res.append([f[0],list(f[1]),0,0])
num_r = 0
while rf and rb:
    nb = rb.pop(0)
    nf = rf.pop(0)
    for b in nb:
        for r in res:
            if r[2] == num_r:
                if b[1][0] == r[1][1] and b[1][1] == r[1][0] and b[1][2] not in r[1]: 
                    tu = r[1].copy()
                    tu.insert(0,b[1][2])
                    res.append([r[0]+b[0],tu,r[2]+1,r[3]])
    for f in nf:
        for r in res:
            if r[3] == num_r:
                if f[1][0] == r[1][-2] and f[1][1] == r[1][-1] and f[1][2] not in r[1]:
                    tu = r[1].copy()
                    tu.append(f[1][2])
                    res.append([r[0]+f[0],tu,r[2],r[3]+1])
    num_r += 1
    
if rb:
    nb = rb.pop(0)
    for b in nb:
        for r in res:
            if r[2] == num_r:
                if b[1][0] == r[1][1] and b[1][1] == r[1][0] and b[1][2] not in r[1]:
                    tu = r[1].copy()
                    tu.insert(0,b[1][2])
                    res.append([r[0]+b[0],tu,r[2]+1,r[3]])

res.sort(reverse = True)

with open(name + ".txt",'w') as box:
    for i in range(len(res)):
        box.write(str(i)+" :"+str(res[i][1]))
        box.write("\n")


# res_len = 20 if len(res) > 20 else len(res)
# for i in range(res_len):
#     print(res[i])
