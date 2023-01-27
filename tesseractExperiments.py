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
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Create mask
        lower = np.array([0, 0, 183])
        upper = np.array([174, 7, 255])
        mask = cv2.inRange(hsv, lower, upper)

        # Dilate
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
        dilate = cv2.dilate(mask, kernel, iterations=5)

        # Find contours
        cnts = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        x, y, w, h = cv2.boundingRect(cnts[0])

        # Invert original image
        inverted = cv2.bitwise_not(img)

        cv2.imshow("inverted", inverted)
        # Crop inverted image
        namebox = inverted[y:y + h, x:x + w]
        cv2.imshow("namebox", namebox)

        name = pytesseract.image_to_string(namebox, config='-l deu --psm 12')
        print(f"{path} modified: {name}")
        name = pytesseract.image_to_string(img, config='-l deu --psm 12')
        print(f"{path} original: {name}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
