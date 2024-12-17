import os

# Load OpenAI API key from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Validate OpenAI API key
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is not set in environment variables.")
