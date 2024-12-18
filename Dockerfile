# Use a Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the application code and JSON file
COPY . /app

# Copy Google Cloud credentials #MAITE
# COPY credentials.json /app/credentials.json

# Install dependencies
ENV PYHTONUNBUFFERED=1
RUN apt-get update \
  && apt-get -y install tesseract-ocr
RUN pip install --no-cache-dir -r requirements.txt

# Expose the default port for debugging purposes
# EXPOSE 8000

# Command to run the application
CMD uvicorn magic.api.fast:app --host 0.0.0.0 --port $PORT
