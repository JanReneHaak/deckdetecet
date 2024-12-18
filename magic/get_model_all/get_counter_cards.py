import openai

from magic.params import *
from magic.get_model_all.get_card_name import clean_up

# Function to get counter cards
def get_counter_cards(card_name: str):
    '''
    This function will output a list with number_of_counters card names. As an
    input, the card_name is needed. The function utilizes the OpenAI API using
    gpt-4o. The output can then be used in the similarity function to show cards
    with a similar ability/oracle text.
    '''

    if "❌" in card_name:

        return "❌ Your card has no name! Did you use the correct image? ❌"

    # Initialize the OpenAI API client using the OpenAI Key saved in .env file
    openai.api_key = OPENAI_API_KEY

    # Generating the prompt for the OpenAI model
    number_of_counters = 3
    # The prompt is a string that will be used to generate the output
    prompt= f"Output {number_of_counters} cards that are effective against this Magic: The Gathering card {card_name}. Just provide the names in seperate lines without numbers and do not use {card_name} in your answer"
    # Make a request to the API to generate text
    response = openai.chat.completions.create(
        model="gpt-4o",  # Using gpt-4o
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100
    )
    # Cleans the output at converts it to a list
    output = response.choices[0].message.content.splitlines()
    output = [word.strip() for word in output]
    return output
