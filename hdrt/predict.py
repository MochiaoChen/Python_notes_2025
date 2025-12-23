
import joblib
import cv2
import numpy as np
import sys
import os

def predict_character(model, image_path, image_size=(32, 32)):
    """
    Loads a single image, preprocesses it, and predicts the character using the loaded SVC model.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image file not found at '{image_path}'")
        return None, None

    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not read image file at '{image_path}'")
        return None, None

    # Preprocess the image (resize, flatten, normalize)
    img_resized = cv2.resize(img, image_size)
    img_vector = img_resized.flatten()
    img_normalized = np.array(img_vector, dtype="float32") / 255.0
    
    # The model expects a 2D array, so we reshape the single sample
    img_sample = img_normalized.reshape(1, -1)

    # Make prediction
    predicted_label = model.predict(img_sample)[0]
    
    # Get confidence score
    confidence_scores = model.predict_proba(img_sample)
    confidence = np.max(confidence_scores)

    return predicted_label, confidence

if __name__ == "__main__":
    MODEL_PATH = 'svc_model.joblib'
    
    # --- 1. Check for command-line argument ---
    if len(sys.argv) < 2:
        print("Usage: python3 predict.py <path_to_image>")
        sys.exit(1)
    
    image_to_predict = sys.argv[1]

    # --- 2. Load the trained model ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'.")
        print("Please run train.py first to create the model file.")
        sys.exit(1)
        
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # --- 3. Predict and display the result ---
    predicted_label, confidence = predict_character(model, image_to_predict)

    if predicted_label:
        print(f"Predicted Character: '{predicted_label}'")
        print(f"Confidence: {confidence:.2f}")

