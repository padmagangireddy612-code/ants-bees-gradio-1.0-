
# Ants vs Bees AI Classifier

A PyTorch image-classification application using ResNet-18
transfer learning and a Gradio web interface.

## Features

- Upload and preview an image
- Predict ants or bees
- Display class probabilities and confidence
- Run locally or through Docker
- Deploy to a cloud web service

## Model

Architecture: ResNet-18
Task: Binary image classification
Input: RGB images resized to 224 x 224 pixels
Classes: ants and bees

The model uses a pretrained ImageNet ResNet-18 backbone
with a two-output classifier.

## Run locally

Create and activate a Python virtual environment.

Install CPU-only PyTorch and Torchvision, then install
the dependencies from requirements.txt.

Run the application:

python app.py

Open http://127.0.0.1:7860/

## Run with Docker

docker build -t ants-bees-gradio:1.0 .
docker run --rm -p 7860:7860 ants-bees-gradio:1.0

## Limitations

The classifier recognizes only ants and bees.
It is not a general-purpose insect identification system.

Prediction quality depends on the training data and model.
