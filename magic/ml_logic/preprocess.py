from PIL import Image
import pytesseract as py
import logging

# Preprocess the image to extract the card name and set
def preprocessing(image_path: str, module: str):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((400, 600))
    image = get_croped_image(image, module)
    if image is None:
        raise ValueError("Input image is None. Check the file path and format.")
    logging.info(f"Image preprocessed successfully")
    return image

# Get the cropped image according to the specified module
def get_croped_image(image, module):
    # Ensure the image is valid
    if image is None:
        raise ValueError("Input image is None. Check the file path and format.")
    # Define the possible modules
    modules = ["whole", "name", "set", "oracle"]

    # Mask the cropped_card according to the specified module using Pillow crop
    # (left x, smaller y coordinate, right x, larger y coordinate)
    if module in modules:
        if module == "whole":
            return image
        elif module == "name":
            return image.crop((0,0,400,65))
        elif module == "set":
            tmp = image.crop((315, 325, 385, 385))
            text = py.image_to_string(tmp)
            if text == "":
                return image.crop((315, 325, 385, 385))
            return tmp
        elif module == "oracle":
            return image.crop((0,350,400,550))
        logging.info(f"Image cropped successfully")
    else:
        print("❌Module not found. Revisit the possible modules and select an existing one!❌")
        return None
