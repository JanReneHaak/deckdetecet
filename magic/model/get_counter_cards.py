from magic.params import *
from get_card_name import clean_up
import openai


def get_counters(card_name: str):
    '''
    This function will output a list with number_of_counters card names. As an
    input, the card_name is needed. The function utilizes the OpenAI API using
    gpt-4o. The output can then be used in the similarity function to show cards
    with a similar ability/oracle text.
    '''

    # Initialize the OpenAI API client using the OpenAI Key saved in .env file
    openai.api_key = OPENAI_API_KEY

    # Generating the prompt for the OpenAI model
    number_of_counters = 3
    prompt= f"Output {number_of_counters} cards that are effective against this Magic: The Gathering card {card_name} Do not provide more information and do not use {card_name} in your answer"

    # Make a request to the API to generate text
    response = openai.chat.completions.create(
        model="gpt-4o",  # Using gpt-4o
        messages = [{"role": "user", "content": prompt}],
        max_tokens = 100
        )

    # Cleans the output at converts it to a list
    output = clean_up(response.choices[0].message.content).splitlines()
    output = [word.strip().title() for word in output]
    return output
