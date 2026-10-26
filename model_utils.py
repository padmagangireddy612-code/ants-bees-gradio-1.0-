
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# Locate the project directory
BASE_DIR = Path(__file__).resolve().parent

# Path to the saved model checkpoint
MODEL_PATH = BASE_DIR / "saved_model" / "ants_bees_resnet18.pth"

# Run predictions using CPU
DEVICE = torch.device("cpu")


def load_model():
    # Check whether the checkpoint exists
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    # Load the saved checkpoint
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    # Read the class names saved during training
    class_names = checkpoint["class_names"]

    # Read image size and normalization values
    image_size = checkpoint["image_size"]
    normalization_mean = checkpoint["normalization_mean"]
    normalization_std = checkpoint["normalization_std"]

    # Re-create the ResNet-18 architecture
    model = models.resnet18(weights=None)

    # Replace the final layer with two output classes
    number_of_features = model.fc.in_features
    model.fc = nn.Linear(number_of_features, len(class_names))

    # Extract the actual trained model weights
    state_dict = checkpoint["model_state_dict"]

    # Load the trained weights into the model
    model.load_state_dict(state_dict)

    # Move model to CPU and enable evaluation mode
    model.to(DEVICE)
    model.eval()

    return model, class_names, image_size, normalization_mean, normalization_std


# Load the model and its preprocessing configuration once
model, CLASS_NAMES, IMAGE_SIZE, NORMALIZATION_MEAN, NORMALIZATION_STD = load_model()


# Define image preprocessing
image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=NORMALIZATION_MEAN,
        std=NORMALIZATION_STD
    ),
])


def predict_image(image: Image.Image):
    # Validate uploaded image
    if image is None:
        raise ValueError("Please upload an image.")

    # Convert image to RGB
    image = image.convert("RGB")

    # Preprocess image and add batch dimension
    input_tensor = image_transform(image).unsqueeze(0)

    # Run inference without calculating gradients
    with torch.inference_mode():
        outputs = model(input_tensor)

        # Convert raw scores into probabilities
        probabilities = torch.softmax(outputs, dim=1)[0]

    # Return class names and their probabilities
    results = {
        CLASS_NAMES[index]: float(probabilities[index].item())
        for index in range(len(CLASS_NAMES))
    }

    return results
