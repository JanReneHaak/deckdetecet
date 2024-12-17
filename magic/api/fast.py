import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import JSONResponse
from pathlib import Path
import shutil
import logging
import os
from sentence_transformers import SentenceTransformer

# Custom imports from the magic package
from magic.utils import logger
from magic.ml_logic.preprocess import preprocessing
from magic.get_model_all.get_card_name import get_card_name
from magic.get_model_all.get_card_set import get_card_set
from magic.get_model_all.get_similar_cards import get_similar_cards
from magic.get_model_all.get_counter_cards import get_counter_cards

# Initialize FastAPI app
app = FastAPI()

# Paths and storage - PERHAPST MOVE TO PARAMS.PY
DATA_PATH = Path(__file__).resolve().parent / "data"
UPLOAD_DIR = Path(DATA_PATH) / "uploaded_image"
CSV_ID = Path(DATA_PATH) / "df_relevant_magic_cards.csv"
CSV_NAME = Path(DATA_PATH) / "df_embedded_magic_cards.csv"

# Preload the CSV ID file
df= pd.read_csv(CSV_ID)
df_emb = pd.read_csv(CSV_NAME)

# Preprocess the embeddings column
def string_to_array(s):
    cleaned = s.replace('[', '').replace(']', '').replace('\n', '')
    return np.array(cleaned.split(), dtype=float)
df_emb['embeddings'] = df_emb['embeddings'].apply(string_to_array)

# Preload the SentenceTransformer model for similar cards search
model_similar = SentenceTransformer("magic/get_model_all/sentence_transformer")

# Health check endpoint to ensure the API is running
@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "Magic API is running on Black Lotus!"}

# Endpoint to process a card image and return card details
@app.post("/get_infos")
def get_infos(image: UploadFile = File(...)):

    # Validate file type
    if not image.content_type.startswith("image"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")
    logging.info(f"Uploaded image: {image.filename}")

    # Save the uploaded image
    try:
        # Save the image to the upload directory
        file_path = UPLOAD_DIR / image.filename
        # Save the image to the file path
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        logging.info(f"Image saved to: {file_path}")
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        raise HTTPException(status_code=500, detail="Image could not be saved.")

    # Generating the response list. All responses from below will be stored in there
    response = []

    # Process the image
    try:
        # Get card name
        card_name = get_card_name(str(file_path), df)
        logging.info(f"Card name: {card_name}")
        if card_name.lower().startswith("❌"):
            raise ValueError("Card name could not be extracted.")

        # Get card set
        # card_set = get_card_set(str(file_path), card_name, df)  # Call the function with arguments
        # logging.info(f"Card set: {card_set}")
        # if card_set.lower().startswith("❌"):
        #   raise ValueError("Card set could not be extracted.")

        # Search card data using name
        card_data = df[df["name"].str.lower() == card_name.lower()].iloc[0]
        # Search card data using name and set
        # card_data = df[(df["name"].str.lower() == card_name.lower()) &
        #               (df["set"].str.lower() == card_set.lower())].iloc[0]
        response.append({
            "name": card_data["name"],
            "set": card_data["set"],
            "image_uri_normal": None if pd.isna(card_data["image_uri_normal"]) else card_data["image_uri_normal"]
        })
        logging.info(f"Card details: {response}")

    except Exception as e:
        logger.error(f"Error processing card: {e}")
        raise HTTPException(status_code=500, detail="Error processing the card.")


    """Fetch similar cards using get_similar_cards function and return their image URLs."""
    try:
        # Call the get_similar_cards function and log the result
        similar_cards = get_similar_cards(card_name, df_emb, model_similar)
        logging.info(f"Similar cards: {similar_cards}")
        # Validate the response
        if not similar_cards:
            logging.error(f"No similar cards found for '{card_name}'.")
            raise ValueError(f"No similar cards found for '{card_name}'.")

        # Set the count for the counter for loop to 1
        count = 1

        # Prepare the response with image URLs
        for name in similar_cards:
            card_row = df[df["name"].str.lower() == name.lower()]
            logging.info(f"Counter card row: {card_row}")
            if not card_row.empty:
                card_data = card_row.iloc[0]  # Take the first match if multiple
                # Append card details to response
                response.append({
                                f"similar_{count}": card_data["name"],
                                f"similar_{count}_uri_normal": None if pd.isna(card_data["image_uri_normal"]) else card_data["image_uri_normal"]
                                })
                logging.info(f"Similar card details: {card_row.iloc[0]}")
            count += 1 # Increase the count every step
    except Exception as e:
        logging.error(f"Error fetching similar cards: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}")


    """Fetch counter cards using OpenAI API and return their image URLs."""
    try:
        #df = pd.read_csv(CSV_ID)
        # Get counter cards
        counter_card_names = get_counter_cards(card_name)
        logging.info(f"Counter cards: {counter_card_names}")
        if isinstance(counter_card_names, str):
            raise ValueError(counter_card_names)

        # Set the count for the counter for loop to 1
        count = 1

        # Fetch card details based on name
        for counter_name in counter_card_names:
            card_row = df[df["name"].str.lower() == counter_name.lower()]
            logging.info(f"Counter card row: {card_row}")
            if not card_row.empty:
                card_data = card_row.iloc[0]  # Take the first match if multiple
                # Append card details to response
                response.append({
                            f"counter_{count}": card_data["name"],
                            f"counter_{count}_uri_normal": None if pd.isna(card_data["image_uri_normal"]) else card_data["image_uri_normal"]
                                })
                logging.info(f"Counter card details: {card_data}")
            count += 1 # Increase the count each iteration
        if not response:
            raise ValueError("No counter cards found in the database.")
        logging.info(f"Response: {response}")

    except Exception as e:
        logger.error(f"Error fetching counter cards: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching counter cards: {e}")

    # Return the final response as a JSON
    return JSONResponse(content=response)
