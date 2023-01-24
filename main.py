# This is a sample Python script.
import cv2
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    img = cv2.imread('c:/temp/teamsdemo.webp')
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #
    # thresh_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    #
    # # Blur the image
    # blur = cv2.GaussianBlur(thresh_inv, (1, 1), 0)
    #
    # thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # cv2.imshow("thresh", thresh)
    #
    # # find contours
    # contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    #
    # mask = np.ones(img.shape[:2], dtype="uint8") * 255
    #
    # cnt = contours[4]
    # cv2.drawContours(img, cnt, -1, (0, 0, 0), 3)
    #
    # # for c in contours:
    # #     get the bounding rect
    #     # x, y, w, h = cv2.boundingRect(c)
    #     # if w * h > 300:
    #     #     cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 0, 255), -1)
    # #
    # # res_final = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
    #
    # # cv2.imshow("boxes", mask)
    # # cv2.imshow("final image", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Display original image
    # cv2.imshow('Original', img)
    # cv2.waitKey(0)

    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # Sobel Edge Detection
    sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
    sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
    sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)  # Combined X and Y Sobel Edge Detection

    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)  # Canny Edge Detection
    # Display Canny Edge Detection Image
    # cv2.imshow('Canny Edge Detection', edges)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Size of the name box: 89*23

    nameBoxContours = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # if cv2.contourArea(c) > 10:
        if 85 < w < 120 and 15 < h < 35:
            nameBoxContours.append(c)


    image_copy = img.copy()
    # draw the contours on a copy of the original image
    cv2.drawContours(image_copy, nameBoxContours, -1, (0, 255, 0), 2)
    print(len(contours), "objects were found in this image.")

    cv2.imshow("contours", image_copy)
    cv2.waitKey(0)



    cv2.waitKey(0)

    cv2.destroyAllWindows()


    # assert img is not None, "file could not be read, check with os.path.exists()"
    # imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    # im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnt = contours[4]
    # cv2.drawContours(img, [cnt], 0, (0,255,0), 3)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
