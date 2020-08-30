from __future__ import division

import json
import os
import sys
import time

import cv2
import numpy as np

VIDEO_PATH = sys.argv[1]

FRAME_NUM = 60

VIDEO_NAME = str(VIDEO_PATH).split("/")[-1].split(".")[0]

phone_res = "results/s2/phone/"

frame_path = "results/s2/frames/"

isdetect = False 

i = 0
ex = 100

protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
nPoints = 22
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

fingers = []
while i<FRAME_NUM:
    pic_name = VIDEO_NAME+"_"+str(i)
    print(pic_name)
    frame = cv2.imread(frame_path+pic_name+".jpg")
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    #use new yolo modle
    with open(phone_res+pic_name+".txt",'r') as f:
        phone = f.readline()[1:-1].split()

    if len(phone) < 4:
        i = i + 1
        continue

    x0 = int(phone[0])-ex if int(phone[0])-ex>0 else 0
    y0 = int(phone[1])-ex if int(phone[1])-ex>0 else 0
    x1 = int(phone[2])+ex if int(phone[2])+ex<frameWidth else frameWidth
    y1 = int(phone[3])+ex if int(phone[3])+ex<frameHeight else frameHeight

    print([y0,y1,x0,x1])
    frame = frame[y0:y1,x0:x1]
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    aspect_ratio = frameWidth/frameHeight
    threshold = 0.1#for keypoint
    t = time.time()

    # input image dimensions for the network
    inHeight = 368
    inWidth = int(((aspect_ratio*inHeight)*8)//8)
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()
    print("time taken by network : {:.3f}".format(time.time() - t))

    # Empty list to store the detected keypoints
    points = []

    for j in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, j, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)


        if prob > threshold :
            # Add the point to the list if the probability is greater than the threshold
            points.append((int(point[0]), int(point[1]),prob))
        else :
            points.append(None)

    print(points)
    if points[8] != None:
        print("Finger is detected in frame"+ str(i))
        isdetect = True
        finger = (i,points[8][0]+x0-10,points[8][1]+y0-10,points[8][2])
        fingers.append(finger)

    i += 1

if isdetect == True:
    result_path = "results/s2/hand/"
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    with open(result_path+VIDEO_NAME+".txt",'w') as box:
        for finger in fingers:
            for a in finger:
                box.write(str(a)+" ")
            box.write("\n")
else:
    with open(result_path+VIDEO_NAME+".txt",'w') as box:
        box.write("none")
