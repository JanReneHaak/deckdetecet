import pandas as pd


def load_card_image_data(json_file):
    """
    Load card image data from a JSON file and extract image URIs and IDs.
    """
    data = pd.read_json(json_file)

    # Handle rows with missing or invalid 'image_uris'
    def extract_image_data(row):
        try:
            return {"id": row["id"], "image_uri": row["image_uris"]["small"]}
        except (TypeError, KeyError):
            return None  # Ignore rows with missing or invalid data

    # Apply extraction and filter out None values
    image_data = data.apply(extract_image_data, axis=1).dropna().tolist()

    return [item for item in image_data if item is not None]
