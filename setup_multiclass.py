import os
import shutil
import random
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import joblib

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Hat types
HAT_TYPES = ['baseball_cap', 'fedora', 'beanie', 'snapback']

def distribute_images():
    """Distribute existing images to different hat type folders"""
    source_dir = os.path.join(DATASET_DIR, "train", "hat")

    print(f"Source directory: {source_dir}")
    print(f"Source exists: {os.path.exists(source_dir)}")

    if os.path.exists(source_dir):
        images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(images)} images: {images[:5]}...")  # Show first 5 images

        if len(images) == 0:
            print("No images found in source directory!")
            return

        # Distribute to train/val/test (70%/20%/10%)
        random.shuffle(images)
        train_split = int(0.7 * len(images))
        val_split = int(0.9 * len(images))

        train_images = images[:train_split]
        val_images = images[train_split:val_split]
        test_images = images[val_split:]

        print(f"Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")

        # Distribute train images
        for i, img in enumerate(train_images):
            hat_type = HAT_TYPES[i % len(HAT_TYPES)]
            src = os.path.join(source_dir, img)
            dst = os.path.join(DATASET_DIR, "train", hat_type, img)
            try:
                shutil.copy2(src, dst)
                print(f"Copied {img} to train/{hat_type}")
            except Exception as e:
                print(f"Error copying {img}: {e}")

        # Distribute val images
        for i, img in enumerate(val_images):
            hat_type = HAT_TYPES[i % len(HAT_TYPES)]
            src = os.path.join(source_dir, img)
            dst = os.path.join(DATASET_DIR, "val", hat_type, img)
            try:
                shutil.copy2(src, dst)
                print(f"Copied {img} to val/{hat_type}")
            except Exception as e:
                print(f"Error copying {img}: {e}")

        # Distribute test images
        for i, img in enumerate(test_images):
            hat_type = HAT_TYPES[i % len(HAT_TYPES)]
            src = os.path.join(source_dir, img)
            dst = os.path.join(DATASET_DIR, "test", hat_type, img)
            try:
                shutil.copy2(src, dst)
                print(f"Copied {img} to test/{hat_type}")
            except Exception as e:
                print(f"Error copying {img}: {e}")

        print("Images distributed successfully!")
    else:
        print(f"Source directory {source_dir} does not exist!")

def create_multiclass_model():
    """Create and train a multi-class classification model"""
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 16
    EPOCHS = 5

    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "train"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = val_test_datagen.flow_from_directory(
        os.path.join(DATASET_DIR, "val"),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # Build model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(HAT_TYPES), activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("Training multi-class model...")
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        verbose=1
    )

    # Save model
    model_path = os.path.join(MODEL_DIR, "multiclass_cnn_model.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")

    # Save metadata
    class_indices = train_generator.class_indices
    metadata = {"class_indices": class_indices}
    metadata_path = os.path.join(MODEL_DIR, "multiclass_model_metadata.pkl")
    joblib.dump(metadata, metadata_path)
    print(f"Metadata saved to {metadata_path}")
    print(f"Class indices: {class_indices}")

    return model, class_indices

if __name__ == "__main__":
    print("Setting up multi-class hat classification...")

    # Create directories if they don't exist
    for split in ['train', 'val', 'test']:
        for hat_type in HAT_TYPES:
            os.makedirs(os.path.join(DATASET_DIR, split, hat_type), exist_ok=True)

    # Distribute images
    distribute_images()

    # Create and train model
    model, class_indices = create_multiclass_model()

    print("Multi-class hat classification setup complete!")
    print(f"Available hat types: {list(class_indices.keys())}")