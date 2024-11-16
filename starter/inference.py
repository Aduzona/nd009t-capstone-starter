import os
import io
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import logging
import json

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def input_fn(request_body, content_type):
    """Deserialize input data from raw image bytes."""
    if content_type == "image/jpeg":
        try:
            logger.info("Processing input as JPEG image.")
            image = Image.open(io.BytesIO(request_body))

            transformation = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            return transformation(image).unsqueeze(0)  # Add batch dimension
        except Exception as e:
            logger.error(f"Error in input_fn: {str(e)}")
            raise ValueError("Failed to process image.")
    raise ValueError(f"Unsupported content type: {content_type}")

def model_fn(model_dir):
    """Load the model from the directory."""
    try:
        logger.info("Loading model.")
        model = models.resnet50(pretrained=False)

        for param in model.parameters():
            param.requires_grad = False

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 5)  # Update for 5 classes
        model_path = os.path.join(model_dir, "model.pth")

        logger.info(f"Loading model state from {model_path}.")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Error in model_fn: {str(e)}")
        raise RuntimeError("Failed to load model.")

''' 
def predict_fn(input_data, model):
    """Run prediction and return class probabilities and indices."""
    try:
        logger.info("Running prediction.")
        with torch.no_grad():
            outputs = model(input_data)  # Raw logits
            probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Convert logits to probabilities
            predictions = torch.argmax(probabilities, dim=1)  # Get class indices
            logger.info(f"Predicted probabilities: {probabilities}")
            logger.info(f"Predicted class: {predictions}")
            return predictions.cpu().numpy().tolist()  # Return as a list of class indices
    except Exception as e:
        logger.error(f"Error in predict_fn: {str(e)}")
        raise RuntimeError("Failed to run prediction.")

''' 
