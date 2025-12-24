import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import joblib
import os
import time

# Path ke model dan metadata
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "cnn_model.h5")

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 3em;
        font-weight: bold;
        color: #4a148c; 
        text-align: center;
        margin-bottom: 20px;
    }

    .sub-header {
        font-size: 1.5em;
        color: #7b1fa2;
        text-align: center;
        margin-bottom: 30px;
    }

    .result-box {
        background-color: #e1bee7; 
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #4a148c;
        margin-top: 20px;
    }

    .no-hat {
        background-color: #fce4ec; 
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #c2185b;
        margin-top: 20px;
    }

    .sidebar .sidebar-content {
        background-color: #ffffff;
    }

    .stButton>button {
        background-color: #4a148c;
        color: white;
        border-radius: 10px;
        font-size: 16px;
        padding: 10px 20px;
        border: none;
    }

    .stButton>button:hover {
        background-color: #6a1b9a;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Fungsi untuk load model berdasarkan pilihan
@st.cache_resource
def load_model_and_metadata(model_name):
    if model_name == "Multiclass CNN":
        model_path = os.path.join(MODEL_DIR, 'multiclass_cnn_model.h5')
        metadata_path = os.path.join(MODEL_DIR, 'multiclass_model_metadata.pkl')
    else:
        model_file = model_name.lower().replace(" ", "_").replace("v2", "").replace("50", "") + "_model.h5"
        model_path = os.path.join(MODEL_DIR, model_file)
        metadata_path = os.path.join(MODEL_DIR, 'model_metadata.pkl')
    
    if not os.path.exists(model_path):
        st.error(f"Model {model_name} tidak ditemukan di {model_path}")
        return None, None
    
    model = load_model(model_path)
    metadata = joblib.load(metadata_path)
    class_indices = metadata['class_indices']
    index_to_class = {v: k for k, v in class_indices.items()}
    return model, index_to_class

# Fungsi untuk load model multiclass (untuk predict jenis topi)
@st.cache_resource
def load_multiclass_model():
    model_path = os.path.join(MODEL_DIR, 'multiclass_cnn_model.h5')
    metadata_path = os.path.join(MODEL_DIR, 'multiclass_model_metadata.pkl')
    if os.path.exists(model_path):
        model = load_model(model_path)
        metadata = joblib.load(metadata_path)
        class_indices = metadata['class_indices']
        index_to_class = {v: k for k, v in class_indices.items()}
        return model, index_to_class
    return None, None

multiclass_model, multiclass_index_to_class = load_multiclass_model()

# Fungsi preprocess gambar
def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# UI Streamlit
st.markdown('<div class="main-header">🎩 Klasifikasi Jenis Topi</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload gambar topi dan dapatkan prediksi jenis topi dengan akurasi tinggi! 🤖</div>', unsafe_allow_html=True)

# Sidebar untuk input
with st.sidebar:
    st.header("⚙️ Pengaturan")
    model_options = ["CNN", "MobileNetV2", "ResNet50", "Multiclass CNN"]
    selected_model = st.selectbox("Pilih Model AI:", model_options, help="Pilih model machine learning untuk klasifikasi.")
    
    uploaded_file = st.file_uploader("📤 Upload Gambar Topi", type=["jpg", "jpeg", "png"], help="Pilih gambar dengan format JPG, JPEG, atau PNG.")
    
    st.info("💡 Tip: Gunakan gambar dengan pencahayaan baik untuk hasil terbaik.")

# Load model berdasarkan pilihan
model, index_to_class = load_model_and_metadata(selected_model)

if model is None:
    st.stop()

if uploaded_file is not None:
    # Tampilkan gambar dan hasil dalam columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Gambar yang Diupload")
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar Topi", use_column_width=True)
    
    with col2:
        st.subheader("🔍 Hasil Prediksi")
        
        # Progress bar saat predict
        with st.spinner("Sedang memproses gambar..."):
            time.sleep(1)  # Simulasi loading
            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)
        
        # Untuk semua model, asumsikan output adalah prob untuk 'hat' (kelas 1)
        prob_hat = predictions[0][0] if predictions.shape[1] == 1 else np.max(predictions[0])  # Untuk multiclass, ambil max
        
        if prob_hat > 0.5:
            is_hat = True
            confidence_hat = prob_hat * 100
        else:
            is_hat = False
            confidence_no_hat = (1 - prob_hat) * 100
        
        if is_hat:
            st.markdown('<div class="result-box">', unsafe_allow_html=True)
            st.success(f"✅ **Ada Topi** (Confidence: {confidence_hat:.2f}%)")
            
            # Predict jenis topi menggunakan Multiclass model
            if multiclass_model is not None:
                with st.spinner("Memprediksi jenis topi..."):
                    multiclass_predictions = multiclass_model.predict(processed_image)
                predicted_index = np.argmax(multiclass_predictions[0])
                predicted_class = multiclass_index_to_class[predicted_index]
                confidence_jenis = multiclass_predictions[0][predicted_index] * 100
                st.info(f"🎓 Jenis Topi: **{predicted_class}** (Confidence: {confidence_jenis:.2f}%)")
                
                # Tampilkan probabilitas semua jenis dalam expander
                with st.expander("📊 Detail Probabilitas Jenis Topi"):
                    for idx, prob in enumerate(multiclass_predictions[0]):
                        class_name = multiclass_index_to_class[idx]
                        st.write(f"{class_name}: {prob * 100:.2f}%")
            else:
                st.warning("Model Multiclass tidak tersedia untuk predict jenis.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="no-hat">', unsafe_allow_html=True)
            st.error(f"❌ **Tidak Ada Topi** (Confidence: {confidence_no_hat:.2f}%)")
            st.markdown('</div>', unsafe_allow_html=True)



