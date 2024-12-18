import logging
from magic.ml_logic.preprocess import preprocessing
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import load_model

# Constants
IMG_SIZE = (224, 224)
MODEL_PATH = "magic/get_model_all/model_sets.h5"
CLASS_NAMES = ['10e', '2ed', '2x2', '2xm', '30a', '3ed', '40k', '4ed', '5dn', '5ed', '6ed', '7ed', '8ed', '9ed', 'a25', 'acr', 'afc', 'akh', 'akr', 'ala', 'all', 'anb', 'apc', 'arb', 'arc', 'atq', 'avr', 'bbd', 'bchr', 'bfz', 'blb', 'blc', 'bng', 'bok', 'brb', 'brc', 'bro', 'brr', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18', 'c19', 'c20', 'c21', 'ced', 'cei', 'chk', 'chr', 'clu', 'cm2', 'cma', 'cmb2', 'cmd', 'cmm', 'cmr', 'cn2', 'cns', 'con', 'csp', 'da1', 'dbl', 'dgm', 'dis', 'dka', 'dmc', 'dmr', 'dmu', 'dom', 'dpa', 'drk', 'dst', 'dtk', 'e01', 'eld', 'ema', 'emn', 'eve', 'exo', 'fbb', 'fdn', 'fem', 'frf', 'fut', 'gk1', 'gk2', 'gn3', 'gpt', 'grn', 'gtc', 'hbg', 'hml', 'hop', 'hou', 'ice', 'iko', 'ima', 'inv', 'isd', 'j21', 'j22', 'j25', 'jmp', 'jou', 'jud', 'khc', 'khm', 'kld', 'ktk', 'lcc', 'lci', 'lea', 'leb', 'leg', 'lgn', 'lrw', 'ltc', 'ltr', 'm10', 'm11', 'm12', 'm13', 'm14', 'm15', 'm19', 'm20', 'm21', 'm3c', 'mat', 'mb2', 'mbs', 'me1', 'me2', 'mh1', 'mh2', 'mh3', 'mic', 'mid', 'mir', 'mkc', 'mkm', 'mm2', 'mm3', 'mma', 'mmq', 'mor', 'mrd', 'mul', 'ncc', 'nec', 'nem', 'neo', 'nph', 'ody', 'ogw', 'onc', 'one', 'ons', 'ori', 'otc', 'otj', 'p02', 'pafr', 'pblb', 'pbro', 'pc2', 'pca', 'pclb', 'pcy', 'pdmu', 'pdom', 'pdsk', 'peld', 'piko', 'pio', 'pip', 'pkhm', 'plc', 'plci', 'pls', 'plst', 'pm20', 'pm21', 'pmkm', 'pmom', 'pneo', 'pone', 'por', 'potj', 'prm', 'psal', 'psnc', 'pstx', 'ptc', 'pthb', 'ptk', 'pvow', 'pwar', 'pwoe', 'pxln', 'pz1', 'pz2', 'ren', 'rix', 'rna', 'roe', 'rtr', 'rvr', 's99', 'scd', 'scg', 'shm', 'sir', 'sld', 'snc', 'soi', 'sok', 'som', 'stx', 'sum', 'td0', 'thb', 'tmp', 'tor', 'tpr', 'tsb', 'tsp', 'tsr', 'uds', 'ulg', 'uma', 'unf', 'unh', 'usg', 'ust', 'vis', 'vma', 'voc', 'vow', 'war', 'wc00', 'wc01', 'wc02', 'wc03', 'wc04', 'wc97', 'wc98', 'wc99', 'who', 'woc', 'woe', 'wot', 'wwk', 'xln', 'zen', 'znr']

# Define ImageNet mean and std values for MobileNetV3
MEAN = [0.485, 0.456, 0.406]  # Mean of ImageNet dataset
STD = [0.229, 0.224, 0.225]   # Std of ImageNet dataset


# Preprocess an image from array format
def preprocess_image(img_array, img_size):
    """Preprocess an image for model prediction."""
    if not isinstance(img_array, np.ndarray):
        raise ValueError("Input must be a NumPy array.")

    img_resized = cv2.resize(img_array, img_size)
    img_resized = img_resized / 255.0  # Normalize
    img_resized = (img_resized - MEAN) / STD  # Standardize
    return np.expand_dims(img_resized, axis=0)

# Define the predict function with updated code that uses the processed image
def get_card_set(image_path, card_name, filtered_df, model):
    try:
        logging.info("Preprocessing image for set prediction...")
        cropped_image = preprocessing(image_path, "set")

        img_array = preprocess_image(np.array(cropped_image), IMG_SIZE)
        predictions = model.predict(img_array)

        # Top 3 predictions
        top_indices = np.argsort(predictions[0])[-3:][::-1]
        top_classes = [CLASS_NAMES[idx] for idx in top_indices]

        # Find a match in the filtered dataframe
        for predicted_set in top_classes:
            if predicted_set in filtered_df['set'].values:
                logging.info(f"Matched set: {predicted_set}")
                return predicted_set

        logging.warning("No matching set found in the database.")
        return "❌ Set could not be identified."

    except Exception as e:
        logging.error(f"Error in get_card_set: {e}")
        return "❌ Set prediction failed."
