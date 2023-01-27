# This is a sample Python script.
import random
import string
import time

import Levenshtein
import cv2
import numpy as np
import pytesseract
import pyautogui
import win32gui
import win32api
from PIL import Image
doShowImages = True


# todo check https://stackoverflow.com/questions/70300189/how-to-keep-only-black-color-text-in-the-image-using-opencv-python

def findMatch(known, text):
    found = [f.lower() for f in text.split() if len(f) >= 3]
    for k in known:
        distances = []
        for i in range(0, min(len(k), len(found))):
            if len(found[i]) >= 3:
                distances.append(Levenshtein.distance(k[i].lower(), found[i].lower()))
        if all(x <= 3 for x in distances):
            print(f"{k} matches {found}")


def findSpeakerNameCoordinates(img):
    # Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # bilateral filter
    blur = cv2.bilateralFilter(gray, 5, 75, 75)
    cv2.imshow("py blur", blur)

    # morphological gradient calculation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
    # cv2.imshow("py grad", grad)

    # binarization
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_OTSU)
    cv2.imshow("py bw", bw)

    # closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("py closed", closed)
    candidateContours = []
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    nameBoxCoordinates = []

    image_copy = img.copy()
    cv2.drawContours(image_copy, contours, -1, (255, 0, 0), 1)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # if 700 < y < 720 and x < 680:
        if 40 < w < 150 and 7 < h < 50:
            cv2.drawContours(image_copy, [c], -1, (0, 255, 0), 1)
            candidateContours.append(c)
            nameBoxCoordinates.append((x, y, w, h))
    cv2.imshow("py contours", image_copy)
    return nameBoxCoordinates


def findSpeakingIndicatorCoordinates(img):
    image_copy = img.copy()
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Create the HSV mask
    lowerValues = np.array([96, 65, 210])
    upperValues = np.array([121, 125, 247])
    hsvMask = cv2.inRange(hsvImage, lowerValues, upperValues)
    cleanedMask = hsvMask
    # Pre-process mask:
    kernelSize = 3
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    iterations = 2
    cleanedMask = cv2.morphologyEx(cleanedMask, cv2.MORPH_DILATE, structuringElement, None, None, iterations,
                                   cv2.BORDER_REFLECT101)
    cleanedMask = cv2.morphologyEx(cleanedMask, cv2.MORPH_ERODE, structuringElement, None, None, iterations,
                                   cv2.BORDER_REFLECT101)
    contours, _ = cv2.findContours(cleanedMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    approxContours = []
    indicatorCoordinateTuples = []
    boundingBoxes = []
    approxContoursImg = img.copy()
    boundingBoxesImg = img.copy()
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if cv2.contourArea(c) > 1000:
            contoursPoly = cv2.approxPolyDP(c, 3, True)
            approxContours.append(contoursPoly)

            # Convert the polygon to a bounding rectangle:
            boundRect = cv2.boundingRect(contoursPoly)
            boundingBoxes.append(boundRect)

            # Get the bounding rect's data:
            rectX = boundRect[0]
            rectY = boundRect[1]
            rectWidth = boundRect[2]
            rectHeight = boundRect[3]
            isWithoutVideo = len(contoursPoly) > 10
            # The shape of the indicator is a circle (indicated by the number of points being more than 4, but 10 to be sure) when there's no video
            # IN that case the name of the speaker is below the circle so we set the rectangle accordingly (roughly)
            if isWithoutVideo:
                rectY = rectY + rectHeight
                rectHeight = 50
            else:
                # The name box is in the lower part of the image
                rectY = rectY + rectHeight - int(rectHeight * 0.1)
                rectHeight = int(rectHeight * 0.1)

            indicatorCoordinateTuples.append((rectX, rectY, rectWidth, rectHeight))
            cv2.rectangle(boundingBoxesImg, (rectX, rectY),
                          (rectX + rectWidth, rectY + rectHeight), (0, 255, 0), 2)

    cv2.drawContours(approxContoursImg, approxContours, -1, (0, 255, 0), 2)
    return indicatorCoordinateTuples


def getSpeakerName(imagepath):
    pytesseract.pytesseract.tesseract_cmd = r'c:\Program Files\Tesseract-OCR\tesseract.exe'
    img = cv2.imread(imagepath)
    # img = cv2.imread(r'C:\Users\strat\PycharmProjects\teamsDetector\teamscall sharing names.png')
    # img = cv2.imread(r'C:\Users\strat\PycharmProjects\teamsDetector\Meeting in jannik speaking.png')
    speakingCoordinators = findSpeakingIndicatorCoordinates(img)
    if len(speakingCoordinators) == 1:  # todo multiple speakers
        x, y, w, h = speakingCoordinators[0][0], speakingCoordinators[0][1], speakingCoordinators[0][2], \
            speakingCoordinators[0][3]
        speakerBox = img[y:y + h, x:x + w]
        if doShowImages:
            cv2.imshow("speaker box", speakerBox)

        nameBoxCoordinates = findSpeakerNameCoordinates(speakerBox)
        image_copy = speakerBox.copy()
        count = 0
        for coordinates in nameBoxCoordinates:
            # cv2.rectangle(speakerBox, (coordinates[0], coordinates[1]),
            #               (coordinates[0] + coordinates[2], coordinates[1] + coordinates[3]), (0, 255, 0), 2)
            x, y, w, h = coordinates[0], coordinates[1], coordinates[2], coordinates[3],
            nameBoxImage = speakerBox[y:y + h, x:x + w]
            name = pytesseract.image_to_string(nameBoxImage)
            if len(name) > 5:
                count += 1
                return name.strip()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    imagepath = r'C:\Users\strat\PycharmProjects\teamsDetector\jannikspeaking.png'
    name = getSpeakerName(imagepath)
    print(name)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
