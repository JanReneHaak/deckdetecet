import numpy as np
from PIL import Image
import cv2
import heapq


def preprocessing(image_path: str, module: str):
    '''
    A function that combines the get_contour and get_card function. Takes the
    image path (or image, depends on how we will process this later on) and returns
    the cropped image that contains only the region of interest, depending on the
    module name.
    '''
    modules = ["whole", "name", "set", "oracle"]

    image = cv2.imread(image_path)
    contour = get_contour(image)
    cropped_image = card_extraction(image, contour)
    cropped_image_contiguous = np.ascontiguousarray(cropped_image)
    image = Image.fromarray(cropped_image_contiguous)
    cropped_image = image.resize((400, 600))

    if module in modules:

        if module == "whole":
            return cropped_image

        elif module == "name":
            return cropped_image.crop((0,0,400,85))

        elif module == "set":
            return cropped_image.crop((0,300,400,375))

        elif module == "oracle":
            return cropped_image.crop((0,350,400,550))

    else:

        print("❌Module not found. Revisit the possible modules and select an existing one!❌")
        return None


def get_contour(image):
    '''
    This function will take an input image and detect different contours
    using Open CV.
    '''

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Applying binary thresholding
    ret, thresh = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

    # Detect the contours on the binary image using cv2.CHAIN_APPROX_SIMPLE
    # Using the SIMPLE method reduces computational time while maintaining
    # accuracy
    contours, _ = cv2.findContours(image=thresh, mode=cv2.RETR_TREE,
                                            method=cv2.CHAIN_APPROX_SIMPLE)

    return contours


def card_extraction(image, contours):
    '''
    This function will take the input image and the contours from above to
    slice the original image based on the second largest contour.
    This will be the card that needs to be detected.
    '''

    areas = []

    # Store contour areas and the corresponding indices
    for index, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        areas.append(((w * h), index))

    # Extract the second largest areas from the areas list
    largest_two = heapq.nlargest(2, {area for area, _ in areas})  # Get the two largest unique areas
    if len(largest_two) > 1:
        second_largest_area = largest_two[1] # Define that the second largest area is the second element

    # Find the contour associated with the second largest area
    for area, index in areas:
        if area == second_largest_area:
            area_key = index
            break

    x, y, w, h = cv2.boundingRect(contours[area_key]) # The index of the second largest contour is used to index the contours

    # Crop the region of interest from the original image
    cropped = image[y:y+h, x:x+w]

    return cropped
