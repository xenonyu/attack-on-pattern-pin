from typing import List, NoReturn
import warnings
import csv
import os
import sys
import cv2
import numpy as np

warnings.filterwarnings("ignore")


class S1TrackDoubleHand:
    def __init__(self):
        self.baseDir = os.getcwd()
        self.batchName = "batch_5"
        self.resultDir = os.path.join(self.baseDir, "results", self.batchName)
        self.dataDir = os.path.join(self.baseDir, "data", self.batchName)
        self.cornerDir = os.path.join(self.resultDir, "corner")
        self.handDir = os.path.join(self.resultDir, "double_hand")
        self.detectRes = {}
        self.startNum = 0
        self.handThreshold = 0.3
        self.checkNum = 60
        self.scale = 2
        self.countStatic = 0
        self.moveThreshold = 10
        self.checkMovingScale = 150
        self.onlyOneHand = True

    def __reformatDetection(self, fileName):
        self.startNum = int(open(os.path.join(self.resultDir, fileName, "start2.txt"), "r").readline())
        hands = open(os.path.join(self.handDir, fileName + ".txt")).read().strip().split("\n")
        hands = [hand.split(" ") for hand in hands]
        hands = [list(map(int, hand[0:3])) + [float(hand[3])] for hand in hands]
        mean = np.mean(np.array(hands)[:, 3])
        hands = [hand for hand in hands if hand[3] >= 0.3 or hand[3] > mean]
        if len(hands) != len(set(hands[0])): self.onlyOneHand = False
        self.detectRes = {}
        for i in range(self.checkNum):
            corner = open(os.path.join(self.cornerDir, fileName + "_" + str(i).zfill(2) + ".txt"), 'r').readline()[1:-1].split()
            if len(corner) < 4: continue
            cornerBox = list(map(int, corner))
            handLocations = [hand for hand in hands if hand[0] == i]
            if not handLocations: continue
            self.detectRes[i] = [handLocations, cornerBox]

    def __center(self, box: tuple) -> List[int]:
        return [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]

    def __drawBox(self, frame: np.array, box: tuple) -> NoReturn:
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv2.rectangle(img=frame, pt1=p1, pt2=p2, color=(255, 0, 0), thickness=1)

    def __show(self, frame, fingerBox, cornerBox, fps, startNum) -> NoReturn:
        self.__drawBox(frame, cornerBox)
        self.__drawBox(frame, fingerBox)
        # Display tracker type on frame
        cv2.putText(frame, "CSRT" + " Tracker", (100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        # cv2.imwrite("test.png", frame)

        cv2.imshow("Tracking: " + str(startNum), cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2)))
        cv2.moveWindow("Tracking: " + str(startNum), -10, -10)

    def __check(self, X: List[int], Y: List[int]) -> bool:
        if len(X) > self.checkMovingScale and abs(X[-1] - X[-self.checkMovingScale-1]) <= 3 and abs(Y[-1] - Y[-self.checkMovingScale-1]) <= 3:
            self.countStatic += 1
            print(abs(X[-1] - X[-self.checkMovingScale-1]), Y[-1] - Y[-self.checkMovingScale-1])
        else:
            self.countStatic = 0
        if self.countStatic > 60:
            for _ in range(self.checkMovingScale):
                X.pop()
                Y.pop()
            return False
        elif len(X) > 1 and (abs(X[-1] - X[-2]) > self.moveThreshold or abs(Y[-1] - Y[-2]) > self.moveThreshold):
            X.pop()
            Y.pop()
            return False
        else: return True

    def __writeToFile(self, fileName: str, X: List[int], Y: List[int]) -> NoReturn:
        rows = []
        direction = open(os.path.join(self.resultDir, fileName, "direction.txt")).readline()
        if direction == '1':
            for i in range(len(X)):
                row = {"frame": i, "X": X[i] - X[0], "Y": Y[0] - Y[i]}
                rows.append(row)
        elif direction == '2':
            for i in range(len(X)):
                row = {"frame": i, "X": Y[i] - Y[0], "Y": X[i] - X[0]}
                rows.append(row)
        elif direction == '3':
            for i in range(len(X)):
                row = {"frame": i, "X": X[0] - X[i], "Y": Y[i] - Y[0]}
                rows.append(row)
        elif direction == '4':
            for i in range(len(X)):
                row = {"frame": i, "X": Y[0] - Y[i], "Y": X[0] - X[i]}
                rows.append(row)
        with open(os.path.join(self.resultDir, fileName, "raw_trajectory.csv"), "w") as f:
            rst_csv = csv.DictWriter(f, ["frame", "X", "Y"])
            rst_csv.writeheader()
            rst_csv.writerows(rows)

    def track(self, videoPath: str, cornerBox: List[int], fingerBox: List[int], startNum: int) -> (List[int], List[int]):
        # Read video
        video = cv2.VideoCapture(str(videoPath))
        video.set(cv2.CAP_PROP_POS_FRAMES, startNum - 1)
        # Exit if video not opened.
        if not video.isOpened():
            print("Could not open video")
            sys.exit()
        ok, frame = video.read()
        frame = cv2.resize(frame, (frame.shape[1] * self.scale, frame.shape[0] * self.scale),
                           interpolation=cv2.INTER_AREA)
        cornerBox = tuple(i * self.scale for i in cornerBox)
        fingerBox = tuple(i * self.scale for i in fingerBox)
        # Set up tracker.
        fingerTracker = cv2.TrackerCSRT_create()
        cornerTracker = cv2.TrackerCSRT_create()
        fingerOK = fingerTracker.init(frame, fingerBox)
        cornerOK = cornerTracker.init(frame, cornerBox)
        # init result
        X, Y = [], []
        self.countStatic = 0
        while fingerOK and cornerOK:
            fingerCenter = self.__center(fingerBox)
            cornerCenter = self.__center(cornerBox)
            X.append(fingerCenter[0] - cornerCenter[0])
            Y.append(fingerCenter[1] - cornerCenter[1])
            if not self.__check(X, Y): break

            frameOK, frame = video.read()
            if not frameOK: break
            frame = cv2.resize(frame, (frame.shape[1] * self.scale, frame.shape[0] * self.scale),
                               interpolation=cv2.INTER_AREA)
            timer = cv2.getTickCount()
            # Update tracker
            fingerOK, fingerBox = fingerTracker.update(frame)
            cornerOK, cornerBox = cornerTracker.update(frame)
            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            self.__show(frame, fingerBox, cornerBox, fps, startNum)
            # Exit if ESC pressed
            k = cv2.waitKey(1) & 0xff
            if k == 27: break
        video.release()
        cv2.destroyAllWindows()
        if len(X) < 200 or max(X) - min(X) < 5: return None, None
        else: return X, Y

    def process(self):
        for fileName in os.listdir(self.dataDir):
            if fileName[0:4] != "SSSM": continue
            if fileName != "SSSM-B5-031.mp4": continue

            fileName, fileType = os.path.splitext(fileName)
            self.__reformatDetection(fileName)
            for k, v in self.detectRes.items():
                frame_no = k
                finger = v[0][0]
                fingerBox = [int(finger[1]) + 2, int(finger[2]) + 2, 16, 16]
                corner = v[1]
                cornerBox = [int((int(corner[0]) + int(corner[2])) / 2 - 10), int((int(corner[1]) + int(corner[3])) / 2 - 5), 20, 20]

                X, Y = self.track(os.path.join(self.dataDir, fileName + fileType), fingerBox, cornerBox, k + self.startNum)
                if X is not None: break
                '''如果不成功则'''
                if len(v[0]) == 1:
                    print("The frame is not correct, process next.")
                    continue
                '''有配对的则尝试另一个'''
                finger = v[0][1]
                X, Y = self.track(os.path.join(self.dataDir, fileName), finger, corner, k + self.startNum)
                if X is not None: break
            if X is not None: self.__writeToFile(fileName=fileName, X=X, Y=Y)
            else: print("fail on {fileName}")



if __name__ == '__main__':
    test = S1TrackDoubleHand()
    test.process()
