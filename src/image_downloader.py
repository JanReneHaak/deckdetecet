import os
import aiohttp
import asyncio

async def download_image_async(session, image_id, uri, folder_path):
    try:
        async with session.get(uri) as response:
            response.raise_for_status()
            file_name = f"{image_id}.jpg"  # Use the id as the file name
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "wb") as file:
                while chunk := await response.content.read(1024):
                    file.write(chunk)
            return f"Downloaded: {file_name}"
    except Exception as e:
        return f"Failed to download {uri}: {e}"

async def download_images_async(image_uris, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    async with aiohttp.ClientSession() as session:
        tasks = [
            download_image_async(session, item['id'], item['image_uri'], folder_path) 
            for item in image_uris
        ]
        return await asyncio.gather(*tasks)
