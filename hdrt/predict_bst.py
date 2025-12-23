
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import sys
import os

# --- Configuration ---
MODEL_PATH = 'best_model.keras'
LABELS_PATH = 'best_model_labels.txt'
IMAGE_SIZE = (32, 32)

def predict_character(model, label_names, image_path):
    """
    Loads a single image, preprocesses it, and predicts the character using the loaded Keras model.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return None, None

    try:
        # Load and preprocess the image
        img = tf.keras.utils.load_img(
            image_path, target_size=IMAGE_SIZE, color_mode='rgb'
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch

        # Make prediction
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        predicted_label = label_names[np.argmax(score)]
        confidence = 100 * np.max(score)

        return predicted_label, confidence
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None, None

if __name__ == "__main__":
    # --- 1. Check for command-line argument ---
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <path_to_image>")
        sys.exit(1)
    
    image_to_predict = sys.argv[1]

    # --- 2. Load the trained model and labels ---
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        print(f"Error: Model ('{MODEL_PATH}') or labels ('{LABELS_PATH}') not found.")
        print("Please run train_bst.py first to create the model files.")
        sys.exit(1)
        
    try:
        model = load_model(MODEL_PATH)
        with open(LABELS_PATH, 'r') as f:
            label_names = [line.strip() for line in f.readlines()]
    except Exception as e:
        print(f"Error loading model or labels: {e}")
        sys.exit(1)

    # --- 3. Predict and display the result ---
    predicted_label, confidence = predict_character(model, label_names, image_to_predict)

    if predicted_label:
        print(f"Predicted Character: '{predicted_label}'")
        print(f"Confidence: {confidence:.2f}%")
