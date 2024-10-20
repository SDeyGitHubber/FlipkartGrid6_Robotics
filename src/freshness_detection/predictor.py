# src/freshness_detection/predictor.py
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load the pre-trained model for freshness detection
def load_freshness_model(model_path='Models/weights/best_model.h5'):
    """
    Loads the pre-trained model from the specified path.
    """
    model = load_model(model_path)
    return model

def predict_freshness(image_path, model, preprocessing_fn):
    """
    Predicts the freshness of the input image using the provided model.
    Uses a preprocessing function to prepare the image.
    """
    preprocessed_image = preprocessing_fn(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class, prediction