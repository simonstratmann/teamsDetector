# This is a sample Python script.
import cv2
import numpy as np


# todo check https://stackoverflow.com/questions/70300189/how-to-keep-only-black-color-text-in-the-image-using-opencv-python

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


def findSpeakingIndicatorCoordinates():
    image_copy = img.copy()
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Create the HSV mask
    lowerValues = np.array([96, 65, 210])
    upperValues = np.array([121, 125, 247])
    hsvMask = cv2.inRange(hsvImage, lowerValues, upperValues)
    # cv2.imshow("hsvMask", hsvMask)
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
    # cv2.imshow("cleanedMask", cleanedMask)
    cv2.imshow("cleanedMaskContours", image_copy)
    return indicatorCoordinateTuples


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img = cv2.imread('jannikSpeaking.png')


    # The HSV mask values:

    speakingIndicatorCoordinateTuples = findSpeakingIndicatorCoordinates()
    print(speakingIndicatorCoordinateTuples)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

