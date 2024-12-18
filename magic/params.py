import os
from pathlib import Path

# Load OpenAI API key from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


# Validate OpenAI API key
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is not set in environment variables.")

# Paths and storage - PERHAPST MOVE TO PARAMS.PY
DATA_PATH = Path(__file__).resolve().parent / "api" / "data"
UPLOAD_DIR = Path(DATA_PATH) / "uploaded_image"
CSV_ID = Path(DATA_PATH) / "df_relevant_magic_cards.csv"
CSV_NAME = Path(DATA_PATH) / "df_embedded_magic_cards.csv"
CSV_PRICE = Path(DATA_PATH) / "prices_df.csv"
