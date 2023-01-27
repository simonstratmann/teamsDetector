# This is a sample Python script.

import cv2
import numpy as np
import pytesseract

doShowImages = True

# todo check https://stackoverflow.com/questions/70300189/how-to-keep-only-black-color-text-in-the-image-using-opencv-python


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    fullBoxes = [
        "fullBoxDeanAxel.png",
        "fullBoxDeanJannikVideoBig.png",
        "fullBoxDeanVideo.png",
        "fullBoxHaukeVideo.png",
        "fullBoxMichaelNoVideo.png"
    ]
    pytesseract.pytesseract.tesseract_cmd = r'c:\Program Files\Tesseract-OCR\tesseract.exe'
    for path in fullBoxes:
        img = cv2.imread(path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 218])
        upper = np.array([157, 54, 255])
        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
        dilate = cv2.dilate(mask, kernel, iterations=5)
        cv2.imshow("dilate", dilate)

        # Find contours and filter using aspect ratio
        # Remove non-text contours by filling in the contour
        cnts = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        # cv2.drawContours(dilate, cnts, -1, (255, 0, 0), 1)
        inverted = cv2.bitwise_not(img)
        # cv2.drawContours(inverted, cnts, -1, (255, 0, 0), 1)
        x, y, w, h = cv2.boundingRect(cnts[0])
        namebox = inverted[y:y + h, x:x + w]
        cv2.imshow("inverted", inverted)
        cv2.imshow("namebox", namebox)

        gray = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 150.0, 255.0, cv2.THRESH_BINARY)
        nameboxbw = bw[y:y + h, x:x + w]
        # cv2.imshow("nameboxbw", nameboxbw)

        # Dean works with cropped and inverted and psm > 7

        i = 13
        # name = pytesseract.image_to_string(cv2.imread("fullBoxDeanCroppedInverted.png"), config='-l deu --psm ' + str(i))
        # print(f"{i}: {name}")

        name = pytesseract.image_to_string(nameboxbw, config='-l deu --psm ' + str(i))
        print(f"{path}: {name}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
