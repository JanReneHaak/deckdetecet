import sys 
sys.path.append("/Users/romainvandercam/code/JanReneHaak/deckdetect/deckdetect/magic")

from magic.ml_logic.preprocess import preprocessing, draw_rectangle, get_croped_image
from get_card_name import get_card_name, image_to_text, clean_up, name_based_similarity
from set_retrieval import preprocess_image_from_array, predict_card_set, set_class_name, matching_prediction_name
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pytesseract as py
from tensorflow.keras.models import load_model
import pandas as pd
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import string
from rapidfuzz import fuzz

data = pd.read_json('sets_recognition/default-cards.json')

image_path = "sets_recognition/zen-213-emeria-the-sky-ruin.png"

card_name = get_card_name(image_path, data)

#loading the model 

matched_prediction = matching_prediction_name(image_path, card_name, data)

print("Starting code...")  # Just before preprocessing begins
print(matched_prediction)
print("Preprocessing completed. Moving to prediction...")