import asyncio
from src.image_downloader import download_images_async
from src.data_loader import load_card_image_data


def download_images():
    """
    Downloads images from the provided JSON file to a target folder.
    """
    json_file = "raw_data/card_collections/default-cards.json"
    output_folder = "raw_data/images_small"

    image_uris = load_card_image_data(json_file)

    print(f"Starting download of {len(image_uris)} images...")
    asyncio.run(download_images_async(image_uris, output_folder))
    print("Image downloading complete!")


if __name__ == "__main__":
    download_images()
