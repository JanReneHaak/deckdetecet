import pandas as pd
import numpy as np
from preprocess import preprocessing
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import string
from fuzzywuzzy import process

def get_card_name(image_path):
    image = preprocessing(image_path, "name")
    text = image_to_text(image)
    name = clean_up(text)
    return name


def image_to_text(image):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text


def clean_up(text):
    text = text.strip()
    text_list = [letter for letter in text if not letter.isdigit()]
    for k, v in enumerate(text_list):
        if v in string.punctuation:
            text_list.pop(k)
    first_cleanup = "".join(text_list)
    for char in string.punctuation:
        name = first_cleanup.replace(char, "")
    return name


card_name = get_card_name("raw_data/images/test_15.jpg")
# cards = pd.read_json("raw_data/2024-12-09_default-cards.json")
# card_names = cards["name"].tolist()

# # Perform fuzzy search
# match, score = process.extractOne(card_name, card_names)

# # Get the matching card
# card = cards[cards["name"] == match]
# print(f"Best Match: {match} (Score: {score})")
# print(card["name"])

for char in string.punctuation:
    if char in card_name:
        print("Ey")

print(card_name)
