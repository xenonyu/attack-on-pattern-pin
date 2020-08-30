import os
import sys

from PIL import Image

file = sys.argv[1] #视频路径

name = str(file).split("/")[-1].split(".")[0]

pic = "results/s2/frames/"+name+"_0.jpg"

I = Image.open(pic) 
I.show()  

direction = input("显示手机方向（1：上，2：右，3：下，4：左）：")

result_path = "results/s2/direction/"
if not os.path.exists(result_path):
    os.mkdir(result_path)

with open(result_path+name+".txt",'w') as d:
    d.write(direction)
