import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pytesseract as py

# Initialize global variables
drawing = False  # True if the mouse is pressed
ix, iy = -1, -1  # Initial coordinates
upper_lower_border = []


def preprocessing(image_path: str, module: str):

    image = cv2.imread(image_path)
    cropped = get_croped_image(image, module)

    return cropped


def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing

    # When the left mouse button is pressed
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y  # Record starting point

    # While moving the mouse and holding the button
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            temp_img = param.copy()  # To show the rectangle interactively
            cv2.rectangle(temp_img, (ix, iy), (x, y), (255, 0, 0), 2)
            cv2.imshow("Image", temp_img)

    # When the mouse button is released
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.rectangle(param, (ix, iy), (x, y), (255, 0, 0), 2)
        cv2.imshow("Image", param)
        upper_lower_border.append((ix, iy, x, y))


def get_croped_image(image, module):
    '''
    A function that combines the get_contour and get_card function. Takes the
    image path (or image, depends on how we will process this later on) and returns
    the cropped image that contains only the region of interest, depending on the
    module name.
    '''
    modules = ["whole", "name", "set", "oracle"]
    cv2.imshow("Image", image)

    # Set the mouse callback
    cv2.setMouseCallback("Image", draw_rectangle, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Resize the image to width, height
    cropped_image_contiguous = np.ascontiguousarray(image)
    card_image = Image.fromarray(cropped_image_contiguous)

    # Resize the image to width, height
    card_image = card_image.crop(upper_lower_border[0])
    cropped_card = card_image.resize((400, 600))

    # Mask the cropped_card according to the specified module using Pillow crop
    # (left x, smaller y coordinate, right x, larger y coordinate)
    if module in modules:

        if module == "whole":
            return cropped_card

        elif module == "name":
            return cropped_card.crop((0,0,400,65))

        elif module == "set":
            tmp = cropped_card.crop((0,300,400,375))
            text = py.image_to_string(tmp)
            if text == "":
                return cropped_card.crop((0,500,400,550))
            return tmp

        elif module == "oracle":
            return cropped_card.crop((0,350,400,550))

    else:

        print("❌Module not found. Revisit the possible modules and select an existing one!❌")
        return None

    # Mask the cropped_card according to the specified module using Pillow crop
    # (left x, smaller y coordinate, right x, larger y coordinate)

    if module in modules:

        if module == "whole":
            return resized_card

        elif module == "name":
            return resized_card[0:400,0:65]

        elif module == "set":
            tmp = resized_card[0:400,325:395]
            text = py.image_to_string(tmp)
            if text == "":
                return resized_card[0:400,500:550]
            return tmp

        elif module == "oracle":
            return resized_card[0:400,350:550]

    else:

        print("❌Module not found. Revisit the possible modules and select an existing one!❌")
        return None
