
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
import numpy as np
import os

# --- Configuration ---
IMAGE_SIZE = (32, 32)
BATCH_SIZE = 32
DATA_DIR = 'Img'
MODEL_PATH = 'best_model.keras'
LABELS_PATH = 'best_model_labels.txt'

def build_model(num_classes, data_augmentation):
    """Builds a robust, regularized CNN model."""
    model = models.Sequential()
    
    # --- Data Augmentation and Rescaling ---
    model.add(layers.Input(shape=IMAGE_SIZE + (3,)))
    model.add(data_augmentation)
    model.add(layers.Rescaling(1./255))

    # --- Convolutional Base ---
    # Block 1
    model.add(layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Block 2
    model.add(layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Block 3
    model.add(layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    
    # --- Classifier Head ---
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5)) 
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

if __name__ == "__main__":
    # --- 1. Load Data ---
    print("Loading data...")
    # Create training and validation datasets from the directory
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='rgb' 
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        color_mode='rgb'
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"Found {num_classes} classes: {class_names}")

    # --- 2. Configure Data Augmentation ---
    data_augmentation = models.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    # --- 3. Build and Compile Model ---
    print("Building model...")
    model = build_model(num_classes, data_augmentation)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # --- 4. Define Callbacks ---
    # Stop training when validation loss doesn't improve for 10 epochs
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )
    # Reduce learning rate when a metric has stopped improving
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2,
        patience=5, 
        min_lr=1e-6
    )

    # --- 5. Train the Model ---
    print("\nStarting training...")
    epochs = 100 
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stopping, reduce_lr]
    )

    # --- 6. Evaluate and Save ---
    print("\nEvaluating final model on validation data...")
    loss, acc = model.evaluate(val_ds)
    print(f"Final Validation Accuracy: {acc * 100:.2f}%")

    if acc > 0.95:
        print("Target accuracy reached!")
    else:
        print("Target accuracy of 95% was not reached. Further tuning may be needed.")

    print(f"Saving model to {MODEL_PATH}")
    model.save(MODEL_PATH)

    print(f"Saving label names to {LABELS_PATH}")
    with open(LABELS_PATH, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")

    print("Done.")
