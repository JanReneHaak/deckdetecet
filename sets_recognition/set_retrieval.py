import sys 
sys.path.append("/Users/romainvandercam/code/JanReneHaak/deckdetect/deckdetect/magic")

from magic.ml_logic.preprocess import preprocessing, draw_rectangle, get_croped_image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications import EfficientNetB0
import os
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import PIL.Image as PILImage  # Updated import for PIL.Image
from tensorflow.keras.models import load_model
import pandas as pd

IMG_SIZE = (224, 224)

img_path = 'zen-213-emeria-the-sky-ruin.png'

class_names = ['10e', '2ed', '2x2', '2xm', '30a', '3ed', '40k', '4ed', '5dn', '5ed', '6ed', '7ed', '8ed', '9ed', 'a25', 'acr', 'afc', 'akh', 'akr', 'ala', 'all', 'anb', 'apc', 'arb', 'arc', 'atq', 'avr', 'bbd', 'bchr', 'bfz', 'blb', 'blc', 'bng', 'bok', 'brb', 'brc', 'bro', 'brr', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18', 'c19', 'c20', 'c21', 'ced', 'cei', 'chk', 'chr', 'clu', 'cm2', 'cma', 'cmb2', 'cmd', 'cmm', 'cmr', 'cn2', 'cns', 'con', 'csp', 'da1', 'dbl', 'dgm', 'dis', 'dka', 'dmc', 'dmr', 'dmu', 'dom', 'dpa', 'drk', 'dst', 'dtk', 'e01', 'eld', 'ema', 'emn', 'eve', 'exo', 'fbb', 'fdn', 'fem', 'frf', 'fut', 'gk1', 'gk2', 'gn3', 'gpt', 'grn', 'gtc', 'hbg', 'hml', 'hop', 'hou', 'ice', 'iko', 'ima', 'inv', 'isd', 'j21', 'j22', 'j25', 'jmp', 'jou', 'jud', 'khc', 'khm', 'kld', 'ktk', 'lcc', 'lci', 'lea', 'leb', 'leg', 'lgn', 'lrw', 'ltc', 'ltr', 'm10', 'm11', 'm12', 'm13', 'm14', 'm15', 'm19', 'm20', 'm21', 'm3c', 'mat', 'mb2', 'mbs', 'me1', 'me2', 'mh1', 'mh2', 'mh3', 'mic', 'mid', 'mir', 'mkc', 'mkm', 'mm2', 'mm3', 'mma', 'mmq', 'mor', 'mrd', 'mul', 'ncc', 'nec', 'nem', 'neo', 'nph', 'ody', 'ogw', 'onc', 'one', 'ons', 'ori', 'otc', 'otj', 'p02', 'pafr', 'pblb', 'pbro', 'pc2', 'pca', 'pclb', 'pcy', 'pdmu', 'pdom', 'pdsk', 'peld', 'piko', 'pio', 'pip', 'pkhm', 'plc', 'plci', 'pls', 'plst', 'pm20', 'pm21', 'pmkm', 'pmom', 'pneo', 'pone', 'por', 'potj', 'prm', 'psal', 'psnc', 'pstx', 'ptc', 'pthb', 'ptk', 'pvow', 'pwar', 'pwoe', 'pxln', 'pz1', 'pz2', 'ren', 'rix', 'rna', 'roe', 'rtr', 'rvr', 's99', 'scd', 'scg', 'shm', 'sir', 'sld', 'snc', 'soi', 'sok', 'som', 'stx', 'sum', 'td0', 'thb', 'tmp', 'tor', 'tpr', 'tsb', 'tsp', 'tsr', 'uds', 'ulg', 'uma', 'unf', 'unh', 'usg', 'ust', 'vis', 'vma', 'voc', 'vow', 'war', 'wc00', 'wc01', 'wc02', 'wc03', 'wc04', 'wc97', 'wc98', 'wc99', 'who', 'woc', 'woe', 'wot', 'wwk', 'xln', 'zen', 'znr']

# Define ImageNet mean and std values for MobileNetV3
MEAN = [0.485, 0.456, 0.406]  # Mean of ImageNet dataset
STD = [0.229, 0.224, 0.225]   # Std of ImageNet dataset


# Update the preprocess function to work with an image (not a file path)
def preprocess_image_from_array(img_array, img_size):
    """
    Preprocess an image for prediction: resizing, normalizing, and expanding dimensions.
    Handles both PIL Images and NumPy arrays.
    """
    # Check if the input is a PIL Image, and convert it to a NumPy array if necessary
    if isinstance(img_array, PILImage.Image):  # Updated to use PILImage
        img_array = np.array(img_array)  # Convert PIL Image to NumPy array

    # Validate the input type
    if not isinstance(img_array, np.ndarray):
        raise ValueError("Invalid image format. Input must be a NumPy array or a PIL Image.")

    # Handle color channels if the image is RGB (PIL) and OpenCV expects BGR
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:  # Color image
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Resize the image
    img_resized = cv2.resize(img_array, img_size)

    # Normalize the image (scale to [0, 1] and adjust using ImageNet mean and std)
    img_resized = img_resized / 255.0  # Scale pixel values to [0, 1]
    img_resized = (img_resized - MEAN) / STD  # Normalize based on ImageNet statistics

    # Expand dimensions to simulate a batch size of 1
    img_resized = np.expand_dims(img_resized, axis=0)

    return img_resized

# Define the predict function with updated code that uses the processed image
def predict_card_set(cropped_image):
    """
    Predict the card set based on the cropped image.
    """
    model = load_model('sets_recognition/model_sets.h5')

    img_array = preprocess_image_from_array(cropped_image, IMG_SIZE)
    
    # Assuming `model` is already loaded and available
    predictions = model.predict(img_array)

    # Get the top 3 predicted class indices and their probabilities
    top_3_indices = np.argsort(predictions[0])[-3:][::-1]  # Sort and take the top 3
    top_3_probabilities = predictions[0][top_3_indices]

    # Map the indices to class labels (assuming `class_names` is available)
    top_3_class_labels = [class_names[index] for index in top_3_indices]

    return top_3_class_labels, top_3_probabilities


# Function to return the class name(s) for a given card
def set_class_name(card_name, data):
    card_name = card_name.lower()
    return data[data['name'] == card_name]


# Now, updating the matching function for cropped image
from IPython.display import Image, display

def matching_prediction_name(image_path, card_name, data):
    """
    Use the cropped image to predict the set, then match the prediction with the class names.
    """
    cropped_image = preprocessing(image_path, "set")
    
    relevant_rows = set_class_name(card_name, data)
    
    # Get the top 3 predicted sets
    predicted_sets, predicted_probs = predict_card_set(cropped_image)
    matched_image_urls = []
    all_matching_rows = []  # List to store all matching rows

    for predicted_set in predicted_sets:
        matching_rows = relevant_rows[relevant_rows['set'] == predicted_set]
        
        if not matching_rows.empty:
            all_matching_rows.append(matching_rows)  # Collect matching rows

            # Collect image URLs for matching rows
            for _, row in matching_rows.iterrows():
                if 'normal' in row['image_uris']:
                    matched_image_urls.append(row['image_uris']['normal'])

    # Display the images
    for url in matched_image_urls:
        display(Image(url=url))
    
    # Combine all matching rows into a single DataFrame
    combined_matching_rows = pd.concat(all_matching_rows, ignore_index=True) if all_matching_rows else pd.DataFrame()
    sets_matching_rows = combined_matching_rows['set']
    set_name = sets_matching_rows[0]
    return set_name