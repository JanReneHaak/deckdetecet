import pandas as pd
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import string
from rapidfuzz import fuzz
import logging

# Load the OCR model and tokenizer
from magic.ml_logic.preprocess import preprocessing

# Initialize OCR model once
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")

def get_card_set(image_path: str, card_name: str, df: pd.DataFrame) -> str:
    """
    Extract the card set name from an image, validate it, and ensure the combination
    of card name and set exists in the CSV file.
    """
    try:
        # Step 1: Preprocess the image to crop the set region
        cropped_image = preprocessing(image_path, "set")
        if cropped_image is None:
            raise ValueError("❌ Set region could not be extracted from the image. ❌")

        # Step 2: Use OCR to extract text from the cropped image
        pixel_values = processor(images=cropped_image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        extracted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Step 3: Clean up the extracted text
        cleaned_set = clean_up_set(extracted_text)
        logging.info(f"Extracted set: {cleaned_set}")

        # Step 4: Validate the set name against the CSV
        possible_sets = df[df["name"].str.lower() == card_name.lower()]["set"].str.lower().unique()

        # Match the extracted set with possible sets using fuzzy matching
        matched_set = validate_set_with_fuzz(cleaned_set, possible_sets)
        if not matched_set:
            raise ValueError(f"❌ Could not validate the set for card '{card_name}'. ❌")

        logging.info(f"Validated set: {matched_set}")
        return matched_set.title()

    except Exception as e:
        logging.error(f"Error in get_card_set: {e}")
        return "❌ Set could not be determined. Please check the input image. ❌"


def clean_up_set(text: str) -> str:
    """
    Clean up the OCR extracted text: remove punctuation and normalize case.
    """
    text = text.strip().lower()
    text = "".join(char for char in text if char not in string.punctuation)
    return text


def validate_set_with_fuzz(extracted_set: str, possible_sets: list) -> str:
    """
    Validate the extracted set name using fuzzy matching against possible sets.
    """
    best_match = None
    best_score = 0

    for valid_set in possible_sets:
        score = fuzz.ratio(extracted_set, valid_set)
        if score > best_score and score > 70:  # Threshold for a valid match
            best_match = valid_set
            best_score = score

    return best_match
