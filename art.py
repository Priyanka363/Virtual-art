import cv2
import numpy as np
import os
import time
import HandPathwayModule as hm
import imutils


path = "Resources"
imglist = os.listdir(path)

head = []
for imgpth in imglist:
    img = cv2.imread(f'{path}/{imgpth}')
    head.append(img)

h=head[0]

cap = cv2.VideoCapture(0)
# cap.set(3, 20)
# cap.set(4, 40)

det = hm.handetection(conf_det=0.8)

while True:

    ret, frame = cap.read()
    frame = imutils.resize(frame, width=1000, height=750)
    frame = cv2.flip(frame,1)


    frame = det.hand_find(frame)
    lmlist = det.find_landmark(frame)

    if len(lmlist) != 0:
        xi, yi = lmlist[8][1:]
        xm, ym = lmlist[12][1:]

        fingers = det.fingUp()

        if fingers[0]==1 and fingers[1]==1:
            cv2.rectangle(frame,(xi,yi),(xm,ym),(255,0,0),cv2.FILLED)
            if yi<100:
                if 25<xi<100:
                    h = head[1]
                elif 157<xi<257:
                    h = head[2]
                elif 305<xi<402:
                    h = head[3]
                elif 444<xi<537:
                    h = head[4]
                elif 570<xi<667:
                    h = head[5]
                # elif 732<xi<903:
                #     h = head[6]

            print("SELECTION MODE")

        if fingers[0]==1 and fingers[1]==0:
            #cv2.putText(frame,"Drawing mode",())
            cv2.circle(frame,(xi,yi),10,(255,0,0),cv2.FILLED)
            print("draw")

    frame[0:100,0:1000]=h

    cv2.imshow("Image", frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()