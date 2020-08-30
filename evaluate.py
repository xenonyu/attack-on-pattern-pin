import os
import xlrd
import pandas as pd
import numpy as np
from xlutils.copy import copy
resultPath = os.path.join("总表.csv")
try:
    data = pd.read_csv(resultPath)
except(IOError):
    print("no 总表.csv found.")
print("spearman相关度")
print("距离相关度: {}".format(data['actual distances'].corr(data['standard distances'],method='spearman')))
print("角度相关度: {}".format(data['actual angles'].corr(data['standard angles'],method='spearman')))
print("距离和角度相关度: {}".format(data['actual distances'].corr(data['actual angles'],method='spearman')))
print("kendall相关度")
print("距离相关度: {}".format(data['actual distances'].corr(data['standard distances'],method='kendall')))
print("角度相关度: {}".format(data['actual angles'].corr(data['standard angles'],method='kendall')))
print("距离和角度相关度: {}".format(data['actual distances'].corr(data['actual angles'],method='kendall')))
print("pearson相关度")
print("距离相关度: {}".format(data['actual distances'].corr(data['standard distances'],method='pearson')))
print("角度相关度: {}".format(data['actual angles'].corr(data['standard angles'],method='pearson')))
print("距离和角度相关度: {}".format(data['actual distances'].corr(data['actual angles'],method='pearson')))

# 原始数据
X1 = pd.Series([1, 2, 3, 4, 5, 6])
Y1 = pd.Series([0.3, 0.9, 2.7, 2, 3.5, 5])


def show(x1, y1):
    print('原始位置x 原x 秩次x 排序x 原始位置y 原y 秩次y 排序y 秩次差的平方')


    for i in range(len(x1)):
        xx1 = x1.sort_values();
        yy1 = y1.sort_values()
    ix = x1.index[i]
    ixx = xx1.index[i]
    iy = y1.index[i]
    iyy = yy1.index[i]
    d_2 = (ixx - iyy) ** 2

    print(' {:5} {:10} {:5} {:5} {:10} {:10} {:5} {:5} {:10.2f}'.format(
        ix, x1[i], ixx, xx1[i], iy, y1[i], iyy, yy1[i], d_2))

# 处理数据删除Nan
x1 = X1.dropna()
y1 = Y1.dropna()
n = x1.count()
x1.index = np.arange(n)
y1.index = np.arange(n)

# 分部计算
d = (x1.sort_values().index - y1.sort_values().index) ** 2
dd = d.to_series().sum()

p = 1 - n * dd / (n * (n ** 2 - 1))

# s.corr()函数计算
r = x1.corr(y1, method='spearman')
print(r, p)
print("\n")


#原始数据
height=pd.Series([180,170,160,150,140,130,120,110])
weight=pd.Series([80,75,90,85,70,60,55,65])


# 处理数据
n=weight.count()
df=pd.DataFrame({'a':height,'b':weight})
df1=df.sort_values('b',ascending=False)
df1.index=np.arange(1,df.a.count()+1)
df2=df1.sort_values('a',ascending=False)
s=pd.Series(df2.index)

P=0
for i in s.index:
    P=P+s[s > s[i]].count()
    s=s.drop(i)


# R=（P-(n*(n-1)/2-P))/(n*(n-1)/2)=(4P/(n*(n-1)))-1
R1 =(P - (n * (n - 1) / 2 - P)) / (n * (n - 1) / 2)#0.5714285714285714
print('R1=',R1)
R2= (4*P / (n * (n-1)))-1                               #0.5714285714285714
print('R2=',R2)



#方法2corr函数计算
r1=height.corr(weight,method='kendall')     #0.5714285714285714
print('r1=',r1)
r2=weight.corr(height,method='kendall')    #0.5714285714285714
print('r2=',r2)