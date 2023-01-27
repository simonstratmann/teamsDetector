import cv2
import numpy as np


def merge(images):
    max_width = 0  # find the max width of all the images
    total_height = 0  # the total height of the images (vertical stacking)

    for image in images:
        # open all images and find their sizes
        if image.shape[1] > max_width:
            max_width = image.shape[1]
        total_height += image.shape[0]

    # create a new array with a size large enough to contain all the images
    final_image = np.zeros((total_height, max_width, 3), dtype=np.uint8)

    current_y = 0  # keep track of where your current image was last placed in the y coordinate
    for image in images:
        # add an image to the final array and increment the y coordinate
        final_image[current_y:image.shape[0] + current_y, :image.shape[1], :] = image
        current_y += image.shape[0]
    return final_image
