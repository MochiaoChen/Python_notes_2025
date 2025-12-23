import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import sys

def predict_image(model, label_names, image_path):
    """Loads an image, preprocesses it, and predicts the character."""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return None

    # Load image in grayscale and resize
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image file at '{image_path}'")
        return None

    img_resized = cv2.resize(img, (32, 32))
    
    # Preprocess for the model
    img_processed = np.array(img_resized, dtype="float32") / 255.0
    img_processed = np.expand_dims(img_processed, 0) # Add batch dimension
    img_processed = np.expand_dims(img_processed, -1) # Add channel dimension

    # Make prediction
    predictions = model.predict(img_processed)
    predicted_index = np.argmax(predictions)
    predicted_label = label_names[predicted_index]
    confidence = np.max(predictions)

    return predicted_label, confidence

if __name__ == "__main__":
    # --- Load Model and Labels ---
    MODEL_PATH = 'hdrt_model.h5'
    LABELS_PATH = 'label_names.txt'

    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        print("Error: Model or label file not found.")
        print("Please run the training script first to generate these files.")
        sys.exit(1)

    try:
        model = load_model(MODEL_PATH)
        with open(LABELS_PATH, 'r') as f:
            label_names = [line.strip() for line in f.readlines()]
    except Exception as e:
        print(f"Error loading model or labels: {e}")
        sys.exit(1)

    # --- Get Image Path from Command Line ---
    if len(sys.argv) < 2:
        print("Usage: python3 hdrt.py <path_to_image>")
        # As an example, let's find and predict one image if no path is given
        print("\n--- Example Prediction ---")
        EXAMPLE_IMAGE = 'Img/A/00x1.png' # Defaulting to a known image
        if os.path.exists(EXAMPLE_IMAGE):
             print(f"No image path provided. Running prediction on an example: '{EXAMPLE_IMAGE}'")
             predicted_label, confidence = predict_image(model, label_names, EXAMPLE_IMAGE)
             if predicted_label:
                print(f"\nPredicted Character: '{predicted_label}'")
                print(f"Confidence: {confidence:.2f}")
        else:
            print(f"Example image '{EXAMPLE_IMAGE}' not found. Please provide an image path.")
        sys.exit(0)

    image_path_to_predict = sys.argv[1]

    # --- Predict ---
    predicted_label, confidence = predict_image(model, label_names, image_path_to_predict)
    if predicted_label:
        print(f"Predicted Character: '{predicted_label}'")
        print(f"Confidence: {confidence:.2f}")