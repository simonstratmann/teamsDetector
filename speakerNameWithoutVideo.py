# This is a sample Python script.
import random
import string

import Levenshtein
import cv2
import numpy as np
import pytesseract
from PIL import Image


def findSpeakerNameCoordinates(img):
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
    cv2.drawContours(image_copy, candidateContours, -1, (0, 255, 0), 2)
    cv2.imshow('candidateContours', image_copy)
    return nameBoxCoordinates


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img = cv2.imread(r'C:\Users\strat\PycharmProjects\teamsDetector\nameboxNoVideo.png')

    pytesseract.pytesseract.tesseract_cmd = r'c:\Program Files\Tesseract-OCR\tesseract.exe'
    strings = []

    nameBoxCoordinates = findSpeakerNameCoordinates(img)

    # speakingCoordinators = findSpeakingIndicatorCoordinates(img)
    if len(nameBoxCoordinates) == 1:  # todo multiple speakers
        x, y, w, h = nameBoxCoordinates[0][0] - 5, nameBoxCoordinates[0][1], nameBoxCoordinates[0][2] + 5, \
        nameBoxCoordinates[0][3]
        speakerBox = img[y:y + h, x:x + w]
        # cv2.imshow("speaker box", speakerBox)

        # image_copy = speakerBox.copy()
        # count = 0
        for coordinates in nameBoxCoordinates:
            # cv2.rectangle(speakerBox, (coordinates[0], coordinates[1]),
            #               (coordinates[0] + coordinates[2], coordinates[1] + coordinates[3]), (0, 255, 0), 2)
            x, y, w, h = coordinates[0], coordinates[1], coordinates[2], coordinates[3],
            nameBoxImage = speakerBox[y:y + h, x:x + w]
            name = pytesseract.image_to_string(nameBoxImage)
            cv2.imwrite(f"namebox.jpg", nameBoxImage)
            # count += 1
            print(name)
            cv2.imshow(''.join(random.choice(string.ascii_letters) for i in range(10)), nameBoxImage)

    # cv2.imshow('Name boxes', image_copy)

    cv2.waitKey(0)

    cv2.destroyAllWindows()
