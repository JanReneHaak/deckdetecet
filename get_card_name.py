import pandas as pd
import numpy as np
import pytesseract
from preprocess import preprocessing
import matplotlib.pyplot as plt
import os
from os import listdir
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

## Generating the card dataframe

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")


def get_card_name(processor, model, image):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return text


names = []
folder_dir = "raw_data"
for images in os.listdir(folder_dir):
    # check if the image ends with png
    if (images.endswith(".jpg")):
        path = f"{folder_dir}/{images}"
        image = preprocessing(path, "name")
        tmp = get_card_name(processor, model, image)
        names.append(tmp)

print(names)
