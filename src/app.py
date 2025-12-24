import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import joblib
import os
import requests
import time

# =========================
# KONFIGURASI PATH
# =========================
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================
# GOOGLE DRIVE DIRECT LINKS
# =========================
DRIVE = "https://drive.google.com/uc?export=download&id="

MODEL_URLS = {
    "CNN": DRIVE + "1tdMjEnHYrXPVFGJbzsc3Bmw03wwv5C4E",
    "MobileNetV2": DRIVE + "1UCvFOKOw692SQ7inPoacu87_tJYKH3D7",
    "ResNet50": DRIVE + "19Xcx8DXKNinMS0agGXW7bCl423D5z8IV",
    "Multiclass CNN": DRIVE + "1fbW_CIt_-_mJhe9D25LEnjTBi437bqvQ"
}

METADATA_URLS = {
    "binary": DRIVE + "19qefnflOfyQ4vio-tyOUgiG40xYAoR_7",
    "multiclass": DRIVE + "15IlsKwuJDtEtvBYdpbCb2HxZxvJVLwSF"
}

# =========================
# DOWNLOAD FILE
# =========================
def download_file(url, save_path):
    if not os.path.exists(save_path):
        with st.spinner(f"📥 Mengunduh {os.path.basename(save_path)}..."):
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        st.success(f"✅ {os.path.basename(save_path)} siap")

# =========================
# LOAD MODEL & METADATA
# =========================
@st.cache_resource
def load_model_and_metadata(model_name):
    model_file = f"{model_name.lower().replace(' ', '_')}_model.h5"
    model_path = os.path.join(MODEL_DIR, model_file)

    download_file(MODEL_URLS[model_name], model_path)

    if model_name == "Multiclass CNN":
        meta_path = os.path.join(MODEL_DIR, "multiclass_model_metadata.pkl")
        download_file(METADATA_URLS["multiclass"], meta_path)
    else:
        meta_path = os.path.join(MODEL_DIR, "model_metadata.pkl")
        download_file(METADATA_URLS["binary"], meta_path)

    model = load_model(model_path)
    metadata = joblib.load(meta_path)

    class_indices = metadata["class_indices"]
    index_to_class = {v: k for k, v in class_indices.items()}

    return model, index_to_class

# =========================
# PREPROCESS IMAGE
# =========================
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img = np.array(image) / 255.0
    return np.expand_dims(img, axis=0)

# =========================
# UI STREAMLIT
# =========================
st.set_page_config(page_title="Klasifikasi Topi", page_icon="🎩", layout="centered")
st.title("🎩 Klasifikasi Topi Menggunakan CNN")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    selected_model = st.selectbox(
        "Pilih Model",
        ["CNN", "MobileNetV2", "ResNet50", "Multiclass CNN"]
    )
    uploaded_file = st.file_uploader(
        "Upload Gambar Topi",
        type=["jpg", "jpeg", "png"]
    )

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar Input", use_column_width=True)

    model, index_to_class = load_model_and_metadata(selected_model)

    with st.spinner("🔍 Memprediksi..."):
        img = preprocess_image(image)
        preds = model.predict(img)
        time.sleep(1)

    # Binary / Multiclass handling
    if preds.shape[1] == 1:
        prob = preds[0][0]
        if prob > 0.5:
            st.success(f"✅ Ada Topi ({prob*100:.2f}%)")
        else:
            st.error(f"❌ Tidak Ada Topi ({(1-prob)*100:.2f}%)")
    else:
        idx = np.argmax(preds[0])
        label = index_to_class[idx]
        conf = preds[0][idx] * 100
        st.success(f"🎓 Jenis Topi: **{label}** ({conf:.2f}%)")

        with st.expander("📊 Detail Probabilitas"):
            for i, p in enumerate(preds[0]):
                st.write(f"{index_to_class[i]}: {p*100:.2f}%")
