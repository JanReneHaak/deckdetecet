import pandas as pd  # For working with DataFrames and CSV files
import numpy as np   # For working with numerical arrays
from sklearn.metrics.pairwise import cosine_similarity # For computing cosine similarity
import logging

# Load the CSV ID file globally
def get_similar_cards(card_name: str, df_emb, model_similar):
    try:
        # Ensure the card name exists in the DataFrame
        if card_name.lower() not in df_emb['name'].str.lower().values:
            raise ValueError(f"Card '{card_name}' not found in the database.")
        # Get the embedding of the input card
        input_embedding = df_emb[df_emb['name'].str.lower() == card_name.lower()].iloc[0]['embeddings']
        logging.info(f"Input embedding: {input_embedding}")

        # Compute cosine similarity
        similarities = cosine_similarity([input_embedding], list(df_emb['embeddings']))
        similarities = similarities.flatten()  # Convert to a 1D array
        logging.info(f"Similarities: {similarities}")

        # Add similarity scores to the DataFrame
        df_emb['similarity'] = similarities
        logging.info(f"DataFrame with similarities: {df_emb}")

        # Get top 3 similar cards (excluding the input card itself)
        similar_cards = (
            df_emb[df_emb['name'].str.lower() != card_name.lower()]
            .nlargest(3, 'similarity')['name']
            .tolist()
        )
        logging.info(f"Similar cards: {similar_cards}")
        return similar_cards

    except Exception as e:
        raise ValueError(f"Error in get_similar_cards: {e}")
