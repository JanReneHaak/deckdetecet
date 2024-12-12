import pandas as pd
from magic.ml_logic.preprocess import preprocessing
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import string
from rapidfuzz import fuzz


def get_card_name(image_path: str, df: pd.DataFrame):
    df["name"] = df["name"].apply(lambda x: x.lower())
    image = preprocessing(image_path, "name")
    text = image_to_text(image)
    cleaned_text = clean_up(text)

    if cleaned_text == "":

        return "❌ Your card has no name! Did you use the correct image? ❌"

    card_name = name_based_similarity(cleaned_text, df).title()
    return card_name


def image_to_text(image):
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text


def clean_up(text: str):
    text = text.strip()
    text_list = [letter for letter in text if not letter.isdigit()]
    for k, v in enumerate(text_list):
        if v in string.punctuation:
            text_list.pop(k)
    first_cleanup = "".join(text_list)
    for char in string.punctuation:
        name = first_cleanup.replace(char, "")
    name = name.lower()
    return name


def name_based_similarity(cleaned_text: str, df: pd.DataFrame):

    df['Similarity'] = df['name'].apply(lambda x: fuzz.ratio(x, cleaned_text))
    card_name = df.name.iloc[df['Similarity'].argmax()]
    return card_name
