import numpy as np
import xlrd
from xlutils.copy import copy
import sys
import os

# 打开文件
workbook = xlrd.open_workbook('match.xlsx')
resultDir = "./results/no-end/no-end_results/"
outputFileName = 'no-end-724.xls'

def standard(stringKey):
    arrayKey = np.array(stringKey.split("-->"), np.int)
    return arrayKey

def removeRepeat(inList):
    pairs = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 4, 7], [2, 5, 8], [3, 6, 9], [1, 5, 9], [7, 5, 3], [3, 2, 1], [6, 5, 4], [9, 8, 7], [7, 4, 1], [8, 5, 2],[9, 6, 3], [9, 5, 1]])
    i = 0
    while(i < len(inList)-2):
        for pair in pairs:
            if (inList[i:i+3] == pair).all():
                inList = np.delete(inList, i+1)
                break
        i = i + 1
    return inList

def getResult(filePath):
    try:
        with open(filePath, "r") as f:  # 打开文件
            datas = f.readlines()  # 读取文件
            datas = [np.array(data.split(":")[1][1:-2].split(", "), np.int) for data in datas]
            datas = [removeRepeat(data) for data in datas]
    except:
        return None
    return datas


# 查看工作表
print("sheets：" + str(workbook.sheet_names()))
# 通过文件名获得工作表,获取工作表1
table = workbook.sheet_by_name('无遮挡')
myDict = {}
for i in range(1, len(table.col_values(0))):
    myDict[table.col_values(0)[i]] = [table.col_values(1)[i], table.col_values(2)[i]]
# print(myDict)
# sys.exit()

workbooknew = copy(workbook)
ws = workbooknew.get_sheet(0)
j = 0
for keys,values in myDict.items():
    j = j + 1
    if(values[1] != 1):
            continue
    myDict[keys][0] = removeRepeat(standard(myDict[keys][0]))
    output = 'None'
    results = getResult(os.path.join(resultDir, keys + ".txt"))
    if not results:
        print("{}: {}".format(keys, output))
        ws.write(j, 3, output)
        continue
    for i in range(len(results)):
        result = results[i].tolist()
        real = myDict[keys][0].tolist()
        if result == real:
            output = i+1
            break
    ws.write(j, 3, output)
    print("{}: {}".format(keys, output))

workbooknew.save(outputFileName)