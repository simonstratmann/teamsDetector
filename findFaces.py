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


def findSpeakerNameBoxes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # _, bw_copy = cv2.threshold(gray, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv2.imshow('closed', bw_copy)

    # bilateral filter
    blur = cv2.bilateralFilter(gray, 5, 75, 75)

    # morphological gradient calculation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, kernel)
    # cv2.imshow('grad', grad)

    # binarization
    _, bw = cv2.threshold(grad, 32, 255.0, cv2.THRESH_BINARY_INV)
    cv2.imshow('bw', bw)

    # closing
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    # closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('closed', closed)
    candidateContours = []
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    nameBoxCoordinates = []

    image_copy = img.copy()
    for c in contours:
        # if len(c) == 4:
        # if len(c) == 4:
        x, y, w, h = cv2.boundingRect(c)
        if w > 100 and h > 100:
            print(c)
            candidateContours.append(c)
            nameBoxCoordinates.append((x, y, w, h))
            # cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # # if 700 < y < 720 and x < 680:
        # if 40 < w < 150 and 7 < h < 50:
    # cv2.drawContours(image_copy, candidateContours, -1, (0, 255, 0), 2)
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
        # , flags=cv2.CV_HAAR_SCALE_IMAGE
    )
    for face in faces:
        x, y, w, h = face
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('candidateContours', image_copy)
    print(f"Found {len(faces)} faces")
    return nameBoxCoordinates


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img = cv2.imread(r'C:\Users\strat\PycharmProjects\teamsDetector\teamscall sharing speaker no video.png')
    findSpeakerNameBoxes(img)

    cv2.waitKey(0)

    cv2.destroyAllWindows()
