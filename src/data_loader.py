import pandas as pd

def load_card_image_data(file_path):
    """
    Load card data from a JSON file and return a list of dicts 
    containing id and image URI.
    """
    data = pd.read_json(file_path)
    return data.apply(
        lambda row: {'id': row['id'], 'image_uri': row['image_uris']['small']}, axis=1
    ).tolist()