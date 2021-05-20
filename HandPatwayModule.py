import mediapipe as mp
import cv2
import time

class handetection():
    def __init__(self,mode=False,hand=2,conf_det=0.5,conf_trac=0.5):
        self.mode=mode
        self.hand=hand
        self.conf_det=conf_det
        self.conf_trac=conf_trac

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode,self.hand,
                                        self.conf_det,self.conf_trac)
        self.mpdraw = mp.solutions.drawing_utils

    def hand_find(self,frame,draw=True):
        imgrgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.info = self.hands.process(imgrgb)
        if self.info.multi_hand_landmarks:
            for landmrk in self.info.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(frame, landmrk,
                                               self.mphands.HAND_CONNECTIONS)
        return frame


    def find_landmark(self,frame,handnum=0,draw=True):
        lmlist=[]

        if self.info.multi_hand_landmarks:
            myhand=self.info.multi_hand_landmarks[handnum]
            for id, lm in enumerate(myhand.landmark):
                h, w, c = frame.shape
                pixval_x, pixval_y = int(lm.x * w), int(lm.y * h)
                lmlist.append([id,pixval_x,pixval_y])
                # if draw:
                #     cv2.circle(frame, (pixval_x, pixval_y), 5, (0, 0, 0), 40)

        return lmlist


def main():
    cam = cv2.VideoCapture(0)
    currt, prevt = 0, 0
    det=handetection()
    while True:
        ret, frame = cam.read()
        frame = det.hand_find(frame)
        lmlist = det.find_landmark(frame)
        currt = time.time()
        fps = 1 / (currt - prevt)
        prevt = currt

        cv2.putText(frame, str(int(fps)), (20, 20), cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 0, 0))
        # print("no")
        cv2.imshow("Image", frame)
        cv2.waitKey(1)



if __name__ == "__main__":
    main()