import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil
import logging
from tensorflow.keras.models import load_model
from magic.utils import logger

from magic.ml_logic.preprocess import preprocessing
from magic.get_model_all.get_card_name import get_card_name
from magic.get_model_all.get_card_set import get_card_set
from magic.get_model_all.get_similar_cards import get_similar_cards
from magic.get_model_all.get_counter_cards import get_counter_cards
from magic.get_model_all.get_price_history import price_history
from magic.params import DATA_PATH, UPLOAD_DIR, CSV_ID, CSV_NAME, CSV_PRICE

# Initialize FastAPI app
app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Helper function to convert strings to numpy arrays
def string_to_array(s):
    """Convert string representation of a list to a NumPy array."""
    cleaned = s.replace('[', '').replace(']', '').replace('\n', '')
    return np.array(cleaned.split(), dtype=float)

# Preload data and models
def preload_data(app):
    """Load data and models into app state."""
    try:
        # Load CSVs
        app.state.df = pd.read_csv(CSV_ID)
        app.state.df_emb = pd.read_csv(CSV_NAME)
        app.state.df_price = pd.read_csv(CSV_PRICE)
        logging.info("CSVs loaded successfully.")
        # Normalize and preprocess columns
        app.state.df_price["name"] = app.state.df_price["name"].str.lower()
        app.state.df_emb['embeddings'] = app.state.df_emb['embeddings'].apply(string_to_array)
        logging.info("Data normalized and preprocessed successfully.")
        # Load model
        app.state.model_set = load_model("magic/get_model_all/model_sets.h5")
        logging.info("Data and models preloaded successfully.")
    except Exception as e:
        logging.error(f"Error loading resources: {e}")
        raise RuntimeError(f"Failed to preload data or models: {e}")

# Call preload_datsa
preload_data(app)

# Health check endpoint
@app.get("/")
def health_check():
    """Health check endpoint."""
    return {"status": "Magic API is running successfully: BLACK LOTUS POWER!"}

# Process card endpoint
@app.post("/process_card")
def process_card(image: UploadFile = File(...)):
    """Process a card image and return card details."""
    try:
        # Validate file type
        if not image.content_type.startswith("image"):
            raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

        # Save uploaded image
        file_path = UPLOAD_DIR / image.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        logging.info(f"Image saved to: {file_path}")

        # Access preloaded resources
        df = app.state.df
        df_price = app.state.df_price
        model_set = app.state.model_set
        logging.info("Preloaded resources accessed successfully.")

        # Get card name
        card_name = get_card_name(str(file_path), df)
        if card_name.lower().startswith("❌"):
            raise ValueError("Card name could not be extracted.")
        logging.info(f"Card name: {card_name}")

        # Filter rows for the card name
        filtered_df = df[df['name'].str.lower() == card_name.lower()]
        if filtered_df.empty:
            raise ValueError("Card name not found in the database.")
        logging.info(f"Filtered data for card name: {filtered_df}")

        # Get card set
        card_set = get_card_set(str(file_path), card_name, filtered_df, model_set)
        if card_set.lower().startswith("❌"):
            raise ValueError("Card set could not be extracted.")
        logging.info(f"Card set: {card_set}")

        # Retrieve card details
        card_data = filtered_df[filtered_df["set"].str.lower() == card_set.lower()].iloc[0]
        response = {
            "id": card_data["id"],
            "name": card_data["name"],
            "set": card_data["set"],
            "oracle_text": None if pd.isna(card_data["oracle_text"]) else card_data["oracle_text"],
            "price_usd": None if pd.isna(card_data["price_usd"]) else card_data["price_usd"],
            "image_uri_normal": None if pd.isna(card_data["image_uri_normal"]) else card_data["image_uri_normal"]
        }
        logging.info(f"Card details: {response}")

        # Add price history to the response
        price_history_response = price_history(card_name, card_set, df_price)
        response["price_history_plot"] = price_history_response.body.decode("utf-8")
        logging.info(f"Card details and price history: {response}")
        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Error processing card: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing the card: {e}")

@app.get("/get_similar_cards/{card_name}")
def get_similar_cards_endpoint(card_name: str):
    """Fetch similar cards based on card name."""
    try:
        # Access preloaded resources
        df_emb = app.state.df_emb
        df = app.state.df

        # Get similar cards
        similar_cards = get_similar_cards(card_name, df_emb)
        if not similar_cards:
            raise ValueError(f"No similar cards found for '{card_name}'.")
        logging.info(f"Similar cards: {similar_cards}")

        # Prepare response with image URLs
        response = []
        for name in similar_cards:
            card_row = df[df["name"].str.lower() == name.lower()]
            if not card_row.empty:
                card_data = card_row.iloc[0]
                response.append({
                    "name": card_data["name"],
                    "image_uri_normal": None if pd.isna(card_data["image_uri_normal"]) else card_data["image_uri_normal"]
                })
                logging.info(f"Similar card: {card_data['name']}")
        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Error fetching similar cards: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching similar cards: {e}")

@app.get("/get_counter_cards/{card_name}")
def get_counter_cards_endpoint(card_name: str):
    """Fetch counter cards using OpenAI API and return their image URLs."""
    try:
        # Access preloaded resources
        df = app.state.df

        # Get counter cards
        counter_card_names = get_counter_cards(card_name)
        if isinstance(counter_card_names, str):
            raise ValueError(counter_card_names)
        logging.info(f"Counter cards: {counter_card_names}")

        # Fetch card details based on counter card names
        response = []
        for counter_name in counter_card_names:
            card_row = df[df["name"].str.lower() == counter_name.lower()]
            if not card_row.empty:
                card_data = card_row.iloc[0]
                response.append({
                    "name": card_data["name"],
                    "image_uri_normal": None if pd.isna(card_data["image_uri_normal"]) else card_data["image_uri_normal"]
                })
                logging.info(f"Counter card: {card_data['name']}")
        if not response:
            raise ValueError("No counter cards found in the database.")
        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Error fetching counter cards: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching counter cards: {e}")
