import argparse
import csv
import sys
from collections import deque

import cv2
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".")

# Path of video
file = sys.argv[1]

# Frame number
frame_no = int(sys.argv[2])

name = str(file).split("/")[-1].split(".")[0]


# Read video
video = cv2.VideoCapture(str(file))
 
# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    sys.exit()
 
# Read first frame.
while frame_no > 0:
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    frame_no -= 1

# Box for finger (top-left.x, top-left.y, bottom-right.x, bottom-right.y)
finger_box = cv2.selectROI(frame, False)
print(str(finger_box))

# Box for phone (top-left.x, top-left.y, bottom-right.x, bottom-right.y)
phone_box = cv2.selectROI(frame, False)
print(str(phone_box))

X = []
Y = []

np.array(X)
np.array(Y) 

finger_center = (int(finger_box[0] + finger_box[2] / 2), int(finger_box[1] + finger_box[3] / 2))
phone_center = (int(phone_box[0] + phone_box[2] / 2), int(phone_box[1] + phone_box[3] / 2))
X.append(finger_center[0]-phone_center[0])
Y.append(finger_center[1]-phone_center[1])

# Set up tracker.
# you can use
 
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[7]
 
if int(major_ver) < 3:
    finger_tracker = cv2.Tracker_create(tracker_type)
    phone_tracker = cv2.Tracker_create(tracker_type)
else:
    if tracker_type == 'BOOSTING':
        finger_tracker = cv2.TrackerBoosting_create()
        phone_tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        finger_tracker = cv2.TrackerMIL_create()
        phone_tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        finger_tracker = cv2.TrackerKCF_create()
        phone_tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        finger_tracker = cv2.TrackerTLD_create()
        phone_tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        finger_tracker = cv2.TrackerMedianFlow_create()
        phone_tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        finger_tracker = cv2.TrackerGOTURN_create()
        phone_tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        finger_tracker = cv2.TrackerMOSSE_create()
        phone_tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        finger_tracker = cv2.TrackerCSRT_create()
        phone_tracker = cv2.TrackerCSRT_create()

finger_ok = finger_tracker.init(frame, finger_box)
phone_ok = phone_tracker.init(frame, phone_box)

l = []
while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            break
         
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        finger_ok, finger_box = finger_tracker.update(frame)   
        phone_ok, phone_box = phone_tracker.update(frame)
 
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
 
        # Draw bounding box
        if finger_ok :
            # Tracking success
            p1 = (int(finger_box[0]), int(finger_box[1]))
            p2 = (int(finger_box[0] + finger_box[2]), int(finger_box[1] + finger_box[3]))
            p3 = (int(phone_box[0]), int(phone_box[1]))
            p4 = (int(phone_box[0] + phone_box[2]), int(phone_box[1] + phone_box[3]))
            cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            cv2.rectangle(frame, p3, p4, (255,0,0), 2, 1)
            finger_center = (int(finger_box[0] + finger_box[2] / 2), int(finger_box[1] + finger_box[3] / 2))
            phone_center = (int(phone_box[0] + phone_box[2] / 2), int(phone_box[1] + phone_box[3] / 2))
            # X.append(finger_center[0])
            # Y.append(finger_center[1])
            X.append(finger_center[0]-phone_center[0])
            Y.append(finger_center[1]-phone_center[1])
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

        cv2.imshow("Tracking", frame)

        # Exit if ESC pressed
        k = cv2.waitKey(0) & 0xff
        if k == 32:
            label = 1
        else:
            label = 0

        l.append(label)

x = []
y = []
direction = '4' #input("显示手机方向（1：上，2：右，3：下，4：左）：")

if direction == '1':
    for i in range(len(X)):
        x.append(X[i]-X[0])
        y.append(Y[0]-Y[i])
elif direction == '2':
    for i in range(len(X)):
        x.append(Y[i]-Y[0])
        y.append(X[i]-X[0])
elif direction == '3':
    for i in range(len(X)):
        x.append(X[0]-X[i])
        y.append(Y[i]-Y[0])
elif direction == '4':
    for i in range(len(X)):
        x.append(Y[0]-Y[i])
        y.append(X[0]-X[i])

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
l = {'label':l}

dx = pd.DataFrame(np.array([standardize(d) for d in dx]).T,columns=["dx"+str(i) for i in range(0-n,1+n) if i != 0])
dy = pd.DataFrame(np.array([standardize(d) for d in dy]).T,columns=["dy"+str(i) for i in range(0-n,1+n) if i != 0])
label = pd.DataFrame(l, index=None)

result = label.join(dx)
result = result.join(dy)

result.to_csv("features/"+name+".csv",index=False)

video.release()
cv2.destroyAllWindows()
