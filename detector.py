import cv2
import numpy as np
import time
from imutils.object_detection import non_max_suppression
class Detector():

    def qrDetect(self, img):
        qrDecoder = cv2.QRCodeDetector()
        
        data,bbox, _ = qrDecoder.detectAndDecode(img)
        if len(data)>0:
            print("Decoded Data : {}" + str(data))
            return bbox
        else:
            print("QR Code not detected")
            return False

    def letterDetect(self, img):
        orig = img.copy()
        (H, W) = img.shape[:2]
        (newW, newH) = (args["width"], args["height"])
        rW = W / float(newW)
        rH = H / float(newH)
        img = cv2.resize(img, (newW, newH))
        (H, W) = img.shape[:2]

        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]
        net = cv2.dnn.readNet("frozen_east_text_detection.pb")
        blob = cv2.dnn.blobFromimg(img, 1.0, (W, H),(123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)
        (numRows, numCols) = scores.shape[2:4]
        rects = []
        confidences = []

        for y in range(0, numRows):
            scoresData = scores[0, 0, y]
            xData0 = geometry[0, 0, y]
            xData1 = geometry[0, 1, y]
            xData2 = geometry[0, 2, y]
            xData3 = geometry[0, 3, y]
            anglesData = geometry[0, 4, y]
            for x in range(0, numCols):
                if scoresData[x] < args["min_confidence"]:
                    continue
                (offsetX, offsetY) = (x * 4.0, y * 4.0)
                angle = anglesData[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                h = xData0[x] + xData2[x]
                w = xData1[x] + xData3[x]
                endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
                endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
                startX = int(endX - w)
                startY = int(endY - h)
                rects.append((startX, startY, endX, endY))
                confidences.append(scoresData[x])

        boxes = non_max_suppression(np.array(rects), probs=confidences)
        print(boxes)
        return boxes

    def colourDetect(self, img):
        imgGry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, thrash = cv2.threshold(imgGry, 240 , 255, cv2.CHAIN_APPROX_NONE)
        contours , hierarchy = cv2.findContours(thrash, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        boxes = []
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
                    boxes.append([x,y,x+w,y+h])
        return boxes
