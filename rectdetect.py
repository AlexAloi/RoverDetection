import numpy as np
import cv2

img = cv2.imread('capture.jpg')
imgGry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thrash = cv2.threshold(imgGry, 240 , 255, cv2.CHAIN_APPROX_NONE)
contours , hierarchy = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01* cv2.arcLength(contour, True), True)
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5
    if len(approx) == 4 :
        x, y , w, h = cv2.boundingRect(approx)
        aspectRatio = float(w)/h
        print(aspectRatio)
        if aspectRatio <= 0.95 or aspectRatio >= 1.05:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)

cv2.imshow("Text Detection", img)
cv2.waitKey(0)