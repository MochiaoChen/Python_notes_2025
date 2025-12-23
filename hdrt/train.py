
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# --- 1. Data Loading and Preprocessing ---

def load_data(data_dir, image_size=(32, 32)):
    """
    Loads images and labels, resizes them, and flattens them into vectors.
    """
    images = []
    labels = []
    label_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    for label_name in label_names:
        folder_path = os.path.join(data_dir, label_name)
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, fname)
                # Load image in grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize and flatten the image
                    img_resized = cv2.resize(img, image_size)
                    img_vector = img_resized.flatten()
                    images.append(img_vector)
                    labels.append(label_name)

    # Normalize pixel values to be between 0 and 1
    images = np.array(images, dtype="float32") / 255.0
    labels = np.array(labels)

    return images, labels, label_names

# --- Main Execution ---

if __name__ == "__main__":
    DATA_DIR = 'Img'
    MODEL_PATH = 'svc_model.joblib'
    LABELS_PATH = 'svc_label_names.txt'

    # 1. Load and preprocess data
    print("Loading and preparing data...")
    X, y, label_names = load_data(DATA_DIR)
    print(f"Loaded {len(X)} samples.")

    # 2. Split data into training and testing sets
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Train the SVC model
    print("Training SVC model... This may take a moment.")
    # Using a linear kernel is often a good baseline for high-dimensional data
    # C is the regularization parameter
    model = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    model.fit(X_train, y_train)
    print("Training complete.")

    # 4. Evaluate the model
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {acc * 100:.2f}%")

    # 5. Save the model and labels
    print(f"Saving model to {MODEL_PATH}")
    joblib.dump(model, MODEL_PATH)
    
    print(f"Saving label names to {LABELS_PATH}")
    with open(LABELS_PATH, 'w') as f:
        for name in label_names:
            f.write(f"{name}\n")
            
    print("Done.")
