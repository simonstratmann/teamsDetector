# This is a sample Python script.
import random
import string

import Levenshtein
import cv2
import numpy as np
import pytesseract
from PIL import Image


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


def areaFilter(minArea, inputImage):
    # Perform an area filter on the binary blobs:
    componentsNumber, labeledImage, componentStats, componentCentroids = \
        cv2.connectedComponentsWithStats(inputImage, connectivity=4)

    # Get the indices/labels of the remaining components based on the area stat
    # (skip the background component at index 0)
    remainingComponentLabels = [i for i in range(1, componentsNumber) if componentStats[i][4] >= minArea]

    # Filter the labeled pixels based on the remaining labels,
    # assign pixel intensity to 255 (uint8) for the remaining pixels
    filteredImage = np.where(np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')

    return filteredImage


def findSpeakerNameCoordinates():
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, bw_copy = cv2.threshold(gray, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # bilateral filter
    blur = cv2.bilateralFilter(gray, 5, 75, 75)

    # morphological gradient calculation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)

    # binarization
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('closed', closed)
    candidateContours = []
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    nameBoxCoordinates = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # if 700 < y < 720 and x < 680:
        if 40 < w < 100 and 7 < h < 50:
            candidateContours.append(c)
            nameBoxCoordinates.append((x, y, w, h))
    image_copy = img.copy()
    # cv2.drawContours(image_copy, candidateContours, -1, (0, 255, 0), 2)
    # cv2.imshow('candidateContours', image_copy)
    return nameBoxCoordinates


def findSpeakingIndicatorCoordinates():
    image_copy = img.copy()
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Create the HSV mask
    lowerValues = np.array([15, 118, 211])
    upperValues = np.array([21, 186, 255])
    hsvMask = cv2.inRange(hsvImage, lowerValues, upperValues)
    cv2.imshow("hsvMask", hsvMask)
    # cv2.imshow("Image", hsvImage)
    minArea = 50
    cleanedMask = areaFilter(minArea, hsvMask)
    # Pre-process mask:
    kernelSize = 3
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    iterations = 2
    cleanedMask = cv2.morphologyEx(cleanedMask, cv2.MORPH_DILATE, structuringElement, None, None, iterations,
                                   cv2.BORDER_REFLECT101)
    cleanedMask = cv2.morphologyEx(cleanedMask, cv2.MORPH_ERODE, structuringElement, None, None, iterations,
                                   cv2.BORDER_REFLECT101)
    contours, _ = cv2.findContours(cleanedMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    indicatorContours = []
    indicatorCoordinateTuples = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if cv2.contourArea(c) > 1000:
            indicatorContours.append(c)
            contoursPoly = cv2.approxPolyDP(c, 3, True)

            # Convert the polygon to a bounding rectangle:
            boundRect = cv2.boundingRect(contoursPoly)

            # Get the bounding rect's data:
            rectX = boundRect[0]
            rectY = boundRect[1]
            rectWidth = boundRect[2]
            rectHeight = boundRect[3]
            indicatorCoordinateTuples.append((rectX, rectY, rectWidth, rectHeight))
    cv2.drawContours(image_copy, indicatorContours, -1, (0, 255, 0), 2)
    cv2.imshow("cleanedMask", cleanedMask)
    cv2.imshow("cleanedMaskContours", image_copy)
    return indicatorCoordinateTuples


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img = cv2.imread('jannikSpeaking.png')

    pytesseract.pytesseract.tesseract_cmd = r'c:\Program Files\Tesseract-OCR\tesseract.exe'
    strings = []

    known = [["Jannik", "Schick"],
             [u"Jörn", "Hauke"],
             ["Hauke", "Plambeck"],
             ["Hauke", u"Schäfer"],
             ["Axel", "Miller"],
             ["Axel", "Dehning"],
             ]

    nameBoxCoordinates = findSpeakerNameCoordinates()
    image_copy = img.copy()
    count = 0
    for coordinates in nameBoxCoordinates:
        cv2.rectangle(image_copy, (coordinates[0], coordinates[1]),
                      (coordinates[0] + coordinates[2], coordinates[1] + coordinates[3]), (0, 255, 0), 2)
        x, y, w, h = coordinates[0], coordinates[1], coordinates[2], coordinates[3],
        nameBoxImage = img[y:y + h, x:x + w]
        name = pytesseract.image_to_string(nameBoxImage)
        cv2.imwrite(f"namebox{count}.jpg", nameBoxImage)
        count += 1
        print(name)
        cv2.imshow(''.join(random.choice(string.ascii_letters) for i in range(10)), nameBoxImage)

    cv2.imshow('Name boxes', image_copy)

    cv2.waitKey(0)

    cv2.destroyAllWindows()
