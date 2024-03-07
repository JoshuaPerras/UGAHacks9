import cv2
import numpy as np
import os
import HandTrackingModule as htm

##################
brushThickness = 10
eraserThickness = 50



##################

folderPath = "Header"
myList = os.listdir(folderPath)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[0]
drawColor = (255, 0, 255)

cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

detector = htm.handDetector(detectionCon=0.85)
xp, yp = 0,0

imgCanvas = np.zeros((480, 640, 3), np.uint8)

while True:
    # import the image
    seccess, img = cap.read()
    img = cv2.flip(img, 1)

    # find hand landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) !=0:

        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]


        # check which fingers are up
        fingers = detector.fingersUp()
        

        # if selection mode - two fingers up
        if fingers[1] and fingers[2]:
            if y1 < 80:
                if 140 <x1<205:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 280 <x1<350:
                    header = overlayList[1]
                    drawColor = (255, 200, 0)
                elif 400 <x1<450:
                    header = overlayList[2]
                    drawColor = (108, 255, 186)
                elif 530 <x1<610:
                    header = overlayList[3]
                    drawColor = (0,0,0)

            cv2.rectangle(img, (x1, y1 - 20), (x2, y2 + 20), drawColor, cv2.FILLED)
            xp, yp = x1, y1



        # if draw mode - Index is up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 10, drawColor, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0,0,0):
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness, cv2.FILLED)
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness, cv2.FILLED)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness, cv2.FILLED)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness, cv2.FILLED)


            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)


    # Setting the header image
    img[0:80, 0:640] = header
    #img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Image", img)

    cv2.waitKey(1)