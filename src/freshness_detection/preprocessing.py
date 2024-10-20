from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

def preprocess_image(image_path, target_size=(256, 256)):
    """
    Preprocesses the uploaded image for freshness detection.
    - Converts image to RGB.
    - Resizes the image.
    - Normalizes the image.
    """
    image = Image.open(image_path)
    
    # Ensure the image is in RGB format (this removes the alpha channel if it exists)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize and preprocess the image
    image = image.resize(target_size)
    image = img_to_array(image)  # Convert to array
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image
