import os
import sys

import cv2

dir = sys.argv[1]

num = sys.argv[3]

target = sys.argv[2]

for root, dirs, files in os.walk(dir):
    for file in files:
        file_name = str(file).split(".")[0]
        file_path = os.path.join(root,file)
        video = cv2.VideoCapture(str(file_path))

        if not video.isOpened():
            print("Could not open video")
            continue


        i = 0

        while i <int(num):
            ok, frame = video.read()
            if not ok:
                print('Cannot read video file')
                break

            pic_name = file_name +"_"+str(i)+".jpg"
            pic_path = os.path.join(target,pic_name)
            cv2.imwrite(pic_path, frame)
            i = i+1

        video.release()
