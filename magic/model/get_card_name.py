import pandas as pd
import numpy as np
from ml_logic.preprocess import preprocessing
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import string

def get_card_name(image_path):
    image = preprocessing(image_path, "name")
    text = image_to_text(image)
    name = clean_up(text)

    if name == "":
        return print("❌ We can't process your card at the current stage. Sorry 🫰")
    else:
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
