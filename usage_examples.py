import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib

# Path ke folder model
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'model')

# Fungsi umum untuk preprocess gambar (sesuai dengan training)
def preprocess_image(image_path, img_size=(224, 224)):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(img_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# 1. Penggunaan Model CNN (cnn_model.h5)
def use_cnn_model(image_path):
    model_path = os.path.join(MODEL_DIR, 'cnn_model.h5')
    metadata_path = os.path.join(MODEL_DIR, 'model_metadata.pkl')  # Asumsikan metadata sama untuk semua single-class model

    if not os.path.exists(model_path):
        return "Model CNN tidak ditemukan."

    model = load_model(model_path)
    metadata = joblib.load(metadata_path)
    class_indices = metadata['class_indices']
    index_to_class = {v: k for k, v in class_indices.items()}

    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)

    # Untuk binary classification (ada topi atau tidak)
    predicted_index = int(predictions[0][0] > 0.5)  # Threshold 0.5
    predicted_class = index_to_class.get(predicted_index, 'Unknown')
    confidence = predictions[0][0] if predicted_index == 1 else 1 - predictions[0][0]

    return f"Model CNN - Prediksi: {predicted_class}, Confidence: {confidence:.2f}"

# 2. Penggunaan Model MobileNetV2 (mobilenet_model.h5)
def use_mobilenet_model(image_path):
    model_path = os.path.join(MODEL_DIR, 'mobilenet_model.h5')
    metadata_path = os.path.join(MODEL_DIR, 'model_metadata.pkl')

    if not os.path.exists(model_path):
        return "Model MobileNetV2 tidak ditemukan."

    model = load_model(model_path)
    metadata = joblib.load(metadata_path)
    class_indices = metadata['class_indices']
    index_to_class = {v: k for k, v in class_indices.items()}

    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)

    predicted_index = int(predictions[0][0] > 0.5)
    predicted_class = 'hat' if predicted_index == 1 else 'no hat'
    confidence = predictions[0][0] if predicted_index == 1 else 1 - predictions[0][0]

    return f"Model MobileNetV2 - Prediksi: {predicted_class}, Confidence: {confidence:.2f}"

# 3. Penggunaan Model ResNet50 (resnet_model.h5)
def use_resnet_model(image_path):
    model_path = os.path.join(MODEL_DIR, 'resnet_model.h5')
    metadata_path = os.path.join(MODEL_DIR, 'model_metadata.pkl')

    if not os.path.exists(model_path):
        return "Model ResNet50 tidak ditemukan."

    model = load_model(model_path)
    metadata = joblib.load(metadata_path)
    class_indices = metadata['class_indices']
    index_to_class = {v: k for k, v in class_indices.items()}

    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)

    predicted_index = int(predictions[0][0] > 0.5)
    predicted_class = 'hat' if predicted_index == 1 else 'no hat'
    confidence = predictions[0][0] if predicted_index == 1 else 1 - predictions[0][0]

    return f"Model ResNet50 - Prediksi: {predicted_class}, Confidence: {confidence:.2f}"

# 4. Penggunaan Model Multiclass CNN (multiclass_cnn_model.h5) - Bonus, karena ada di workspace
def use_multiclass_cnn_model(image_path):
    model_path = os.path.join(MODEL_DIR, 'multiclass_cnn_model.h5')
    metadata_path = os.path.join(MODEL_DIR, 'multiclass_model_metadata.pkl')

    if not os.path.exists(model_path):
        return "Model Multiclass CNN tidak ditemukan."

    model = load_model(model_path)
    metadata = joblib.load(metadata_path)
    class_indices = metadata['class_indices']
    index_to_class = {v: k for k, v in class_indices.items()}

    processed_image = preprocess_image(image_path)
    predictions = model.predict(processed_image)

    predicted_index = np.argmax(predictions[0])
    predicted_class = index_to_class[predicted_index]
    confidence = predictions[0][predicted_index]

    return f"Model Multiclass CNN - Prediksi: {predicted_class}, Confidence: {confidence:.2f}"

# Contoh penggunaan
if __name__ == "__main__":
    # Ganti dengan path gambar Anda
    image_path = "path/to/your/hat_image.jpg"

    print("Penggunaan Model CNN:")
    print(use_cnn_model(image_path))

    print("\nPenggunaan Model MobileNetV2:")
    print(use_mobilenet_model(image_path))

    print("\nPenggunaan Model ResNet50:")
    print(use_resnet_model(image_path))

    print("\nPenggunaan Model Multiclass CNN:")
    print(use_multiclass_cnn_model(image_path))