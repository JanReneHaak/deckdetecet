import random
import asyncio
from src.image_downloader import download_images_async
from src.data_loader import load_card_image_data


def download_images(json_file, output_folder, num_images=None):
    """
    Downloads images from the provided JSON file to a target folder.
    """
    # Load all image URIs
    image_uris = load_card_image_data(json_file)

    # If num_images is provided, select a random subset
    if num_images is not None:
        if num_images > len(image_uris):
            print(
                f"Requested {num_images} images, but only {len(image_uris)} are available. Downloading all images."
            )
            num_images = len(image_uris)
        image_uris = random.sample(image_uris, num_images)

    print(f"Starting download of {len(image_uris)} images...")
    asyncio.run(download_images_async(image_uris, output_folder))
    print("Image downloading complete!")


if __name__ == "__main__":
    # Input file and output folder
    json_file = "raw_data/card_collections/default_cards.json"
    output_folder = "raw_data/default_cards_images"

    # 1. Download all images
    # download_images(json_file, output_folder)

    # 2. Download a random subset of N images
    download_images(json_file, output_folder, num_images=30)
