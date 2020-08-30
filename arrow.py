import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

file = "/Users/xym/work/lab/python/attack-on-pattern-pin/results/no-end-dis/SSSM-B3-00047/raw_trajectory.csv"

def arrow(filePath):
    name = filePath.split("/")[-2]
    fileType = filePath.split("/")[-1].split(".")[0]
    data = pd.read_csv(filePath)
    x = data["X"].values
    y = data["Y"].values

    plt.figure(figsize=(10,10), dpi=60)
    ax = plt.subplot(111)
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))


    plt.xlim(x.min()*1.1, x.max()*1.1)

    plt.ylim(y.min()*1.1, y.max()*1.1)


    for i in range(len(x)-1):
        dx = x[i+1] - x[i]
        dy = y[i+1] - y[i]
        head_length = (dx + dy) / 20
#         plt.arrow(x[i],y[i],dx,dy,length_includes_head=True,head_width=head_length,head_length=head_length,overhang=0.6,color = 'blue')
        plt.annotate("", xy=(x[i], y[i]), xytext=(x[i+1], y[i+1]), arrowprops=dict(arrowstyle="<-", color="r"))
        plt.grid()

    figName = filePath.split(".")[0]+".jpg"
    print(figName)
    plt.savefig(figName, format='png', transparent=True, dpi=300, pad_inches = 0)
    plt.show()
arrow(file)