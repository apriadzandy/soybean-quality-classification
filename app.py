import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Klasifikasi Kualitas Kacang Kedelai",
    page_icon="🌱",
    layout="wide",
)

MODELS_DIR = Path("models")  # <- folder models di project
MODEL_PATHS = {
    "Model 1 - Scratch CNN": MODELS_DIR / "cnn_scratch.h5",
    "Model 2 - ResNet50": MODELS_DIR / "resnet50.h5",
    "Model 3 - MobileNetV2": MODELS_DIR / "mobilenetv2.h5",
}
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"
IMG_SIZE = (224, 224)

# =========================
# CSS (biar tampilannya keren & kontras aman)
# =========================
st.markdown(
    """
<style>
/* Background */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(1200px 700px at 10% 10%, #16223a 0%, #0b0f19 45%, #070a12 100%);
}
[data-testid="stHeader"] { background: rgba(0,0,0,0); }

/* Main container spacing */
.main .block-container { padding-top: 2rem; padding-bottom: 2rem; }

/* Cards */
.card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 18px;
    padding: 18px 18px;
    box-shadow: 0 12px 36px rgba(0,0,0,0.35);
}
.title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: 0.3px;
    margin-bottom: 6px;
}
.subtitle {
    color: rgba(255,255,255,0.75);
    margin-top: 0px;
    font-size: 1.02rem;
}

/* Small badge */
.badge {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    background: rgba(99, 102, 241, 0.18);
    border: 1px solid rgba(99, 102, 241, 0.35);
    color: rgba(255,255,255,0.9);
    font-size: 0.86rem;
}

/* Prediction label */
.pred {
    font-size: 1.35rem;
    font-weight: 800;
}
.muted { color: rgba(255,255,255,0.7); }

/* Make file uploader darker */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.05);
    border: 1px dashed rgba(255,255,255,0.22);
    border-radius: 16px;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# Load class names & models
# =========================
@st.cache_resource
def load_class_names():
    if CLASS_NAMES_PATH.exists():
        return json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    # fallback (kalau file hilang)
    return ["Broken soybeans", "Immature soybeans", "Intact soybeans", "Skin-damaged soybeans", "Spotted soybeans"]

@st.cache_resource
def load_model(model_path: Path):
    return tf.keras.models.load_model(model_path)

CLASS_NAMES = load_class_names()

# =========================
# Preprocess utilities
# =========================
def pil_to_array(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    return arr

def preprocess_for(model_key: str, arr: np.ndarray) -> np.ndarray:
    """
    arr shape: (H,W,3) float32 0..255
    return: (1,H,W,3) float32 sesuai preprocess model
    """
    x = np.expand_dims(arr, axis=0)

    if "ResNet50" in model_key:
        x = tf.keras.applications.resnet50.preprocess_input(x)  # -> sekitar -123..151
    elif "MobileNetV2" in model_key:
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # -> -1..1
    else:
        # Scratch CNN: normalize 0..1
        x = x / 255.0

    return x

def predict(model_key: str, img: Image.Image):
    model = load_model(MODEL_PATHS[model_key])
    arr = pil_to_array(img)
    x = preprocess_for(model_key, arr)
    probs = model.predict(x, verbose=0)[0]  # (num_classes,)
    probs = probs.astype(float)

    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else str(idx)
    conf = float(probs[idx])

    df = pd.DataFrame({"Kelas": CLASS_NAMES, "Probabilitas": probs})
    df = df.sort_values("Probabilitas", ascending=False).reset_index(drop=True)
    return label, conf, df

# =========================
# UI
# =========================
st.markdown(
    f"""
<div class="card">
  <div class="title">🌱 Klasifikasi Kualitas Kacang Kedelai</div>
  <div class="subtitle">Upload foto kedelai → pilih model → lihat prediksi & probabilitas.</div>
  <div style="margin-top:10px;">
    <span class="badge">Scratch CNN</span>
    <span class="badge">ResNet50 (Transfer Learning)</span>
    <span class="badge">MobileNetV2 (Transfer Learning)</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

left, right = st.columns([1.05, 1.0], gap="large")

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📥 Input Gambar")
    model_choice = st.selectbox("Pilih Model", list(MODEL_PATHS.keys()))
    uploaded = st.file_uploader("Upload gambar kedelai (jpg/png/webp)", type=["jpg", "jpeg", "png", "webp"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is not None:
        img = Image.open(uploaded)
        st.image(img, caption="Gambar yang diupload", use_container_width=True)
    else:
        img = None

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🔎 Prediksi")
    if img is None:
        st.info("Silakan upload gambar terlebih dahulu.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        try:
            label, conf, df = predict(model_choice, img)

            st.markdown(f"<div class='pred'>{label}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='muted'>Model: <b>{model_choice}</b></div>", unsafe_allow_html=True)
            st.metric("Confidence", f"{conf*100:.2f}%")

            st.write("")
            st.subheader("📊 Probabilitas Kelas")
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Chart
            chart_df = df.set_index("Kelas")[["Probabilitas"]]
            st.bar_chart(chart_df, use_container_width=True)

        except Exception as e:
            st.error("Gagal melakukan prediksi. Cek apakah file model & class_names.json sudah benar.")
            st.exception(e)

        st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.caption(
    "Catatan: web ini hanya mendeteksi kualitas kacang kedelai dari gambar yang di upload."
)
