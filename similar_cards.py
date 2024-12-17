# pip install torch
# pip install transformers

import pandas as pd  # For working with DataFrames and CSV files
import numpy as np   # For working with numerical arrays
import torch         # For using tensor operations, as `model.similarity` likely uses PyTorch
from sentence_transformers import SentenceTransformer



def string_to_array(s):
    # Remove brackets and newlines
    cleaned = s.replace('[', '').replace(']', '').replace('\n', '')
    # Convert to numpy array of floats
    return np.array(cleaned.split(), dtype=float)




def similar_cards(input_string):
    df_emb = pd.read_csv('trim_emb_df4.csv',index_col=0)
    model = SentenceTransformer('saved_model/my_model')

    # Apply the function to the Series
    df_emb['embeddings'] = df_emb['embeddings'].apply(string_to_array)


    result = df_emb[df_emb['name'] == input_string]

    similarities = model.similarity(df_emb['embeddings'],result.iloc[0]['embeddings'])

    # Convert the tensor to a NumPy array
    temp_sim_np_array = similarities.detach().numpy()
    temp_sim_np_array
    # Create a Pandas DataFrame from the NumPy array

    sim_df = pd.DataFrame(temp_sim_np_array)

    count = 0
    card_name = df_emb['name'][sim_df[0].sort_values(ascending=False).index[0]]
    similar_cards = [card_name]
    test = 0
    while count < 5:
        if df_emb['name'][sim_df[0].sort_values(ascending=False).index[test]] not in similar_cards:
            similar_cards.append(df_emb['name'][sim_df[0].sort_values(ascending=False).index[test]])
            count = count + 1
        test = test +1

    return (similar_cards[1:])
