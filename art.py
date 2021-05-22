import cv2
import numpy as np
import os
import time
import HandPathwayModule as hm
import imutils

path = "Resources"
imglist = os.listdir(path)
xprev = 0
yprev = 0
brushth = 10
erserth = 25
head = []

for imgpth in imglist:
    img = cv2.imread(f'{path}/{imgpth}')
    head.append(img)

h=head[0]

cap = cv2.VideoCapture(0)
# cap.set(3, 20)
# cap.set(4, 40)
drawboard = np.zeros((750, 1000, 3), np.uint8)

det = hm.handetection(conf_det=0.8, conf_trac=0.8)
col = (255, 255, 255)

while True:

    ret, frame = cap.read()
    frame = imutils.resize(frame, width=1000, height=750)
    frame = cv2.flip(frame, 1)

    frame, ishand = det.hand_find(frame)
    lmlist = det.find_landmark(frame)

    if len(lmlist) != 0:
        xi, yi = lmlist[8][1:]
        xm, ym = lmlist[12][1:]

        fingers = det.fingUp()

        if fingers[0] == 1 and fingers[1] == 1:
            xprev = 0
            yprev = 0
            if yi < 100:
                if 35 < xi < 100:
                    h = head[1]
                    col = (180, 105, 255)
                elif 145 < xi < 215:
                    h = head[2]
                    col = (0, 0, 255)
                elif 270 < xi < 335:
                    h = head[3]
                    col = (255, 0, 0)
                elif 395 < xi < 460:
                    h = head[4]
                    col = (0, 255, 255)
                elif 500 < xi < 570:
                    h = head[5]
                    col = (0, 165, 255)
                elif 610 < xi < 680:
                    h = head[6]
                    col = (0, 255, 0)
                elif 710 < xi < 780:
                    h = head[7]
                    col = (130, 0, 75)
                elif 825 < xi < 990:
                    h = head[8]
                    col = (0, 0, 0)

            cv2.rectangle(frame,(xi,yi),(xm,ym),col,cv2.FILLED)
            #print("SELECTION MODE")

        if fingers[0] == 1 and fingers[1] == 0:
            # cv2.putText(frame,"Drawing mode",())

            if xprev == 0 and yprev == 0:
                xprev = xi
                yprev = yi

            if col == (0, 0, 0):
                cv2.line(frame, (xprev, yprev), (xi, yi),
                         col, erserth)
                cv2.line(drawboard, (xprev, yprev), (xi, yi),
                         col, erserth)

            elif col == (255, 255, 255):
                pass

            else:
                cv2.line(frame, (xprev, yprev), (xi, yi),
                         col, brushth)
                cv2.line(drawboard, (xprev, yprev), (xi, yi),
                         col, brushth)

            cv2.circle(frame, (xi, yi), 10, col, cv2.FILLED)
            # print("draw")

            xprev = xi
            yprev = yi

            if ishand == False:
                xprev = 0
                yprev = 0

    frame[0:140,0:1000]=h
    #frame = cv2.addWeighted(frame,0.)
    cv2.imshow("Image", frame)
    cv2.imshow("Draw Board", drawboard)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()