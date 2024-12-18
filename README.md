# DeckDetect: Magic: The Gathering Card Detection and Similarity API

**DeckDetect** is a backend service that provides image-based detection and recommendation of Magic: The Gathering cards. It leverages computer vision, NLP, and cloud APIs to detect card names, retrieve their details, find similar cards, and suggest counter cards.

---

## **Features**
- **Card Detection:**
  Upload an image of a Magic: The Gathering card and retrieve card details, such as name, oracle text, price, and set.

- **Card Similarity:**
  Get recommendations for similar cards based on embeddings.

- **Counter Card Suggestions:**
  Use OpenAI's GPT models to suggest counter cards for a given card.


- **Cloud Integration:**
  Integration with Google Cloud for storage and deployment.

---

## **Architecture**
The project is built with:
- **FastAPI:** Backend API framework.
- **Hugging Face Models:** For OCR and embeddings.
- **Google Cloud:** For scalable storage and deployment.
- **Docker:** For containerization and deployment.
- **Python:** Core programming language.


# VS CODE ARCHITECTURE
# Card metadata and uploaded images
magic/api/data/
# FastAPI endpoints
magic/api/
# Preprocessing and utility logic
magic/ml_logic/
# ML models and utilities
magic/models/
# Docker configuration
Dockerfile
# Automation tasks
Makefile
# Python dependencies
requirements.txt
# Package setup
setup.py


# Endpoints

(1) /health_check (GET)
# Description: Healtcheck

(2) /process_card (POST)
# Description: Upload an image to detect name and set of the card and and retrieve card details from a stored csv.

(3) get_similar_cards/{card_name} (GET)
# Description: Retrieve cards similar to the given card name using embeddings and a pre-learnd model from huggingfaces.

(4) /get_counter_cards/{card_name} (GET)
# Description: Retrieve counter cards for a given card using GPT-based suggestions.
