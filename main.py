# Premium GI Image Segmentation Dashboard using U-Net

import os
import io
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, models


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="GI Segmentation AI",
    page_icon="🧠",
    layout="wide"
)


# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f8fbff 0%, #eef4ff 50%, #f9f7ff 100%);
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #1f2937;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    color: #6b7280;
    margin-bottom: 25px;
}

.card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 8px 25px rgba(0,0,0,0.08);
    border: 1px solid #e5e7eb;
    margin-bottom: 20px;
}

.metric-card {
    background: white;
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0 6px 18px rgba(0,0,0,0.07);
    border: 1px solid #e5e7eb;
}

.metric-value {
    font-size: 28px;
    font-weight: 800;
    color: #2563eb;
}

.metric-label {
    font-size: 14px;
    color: #6b7280;
}

.success-box {
    background: #ecfdf5;
    color: #065f46;
    padding: 14px;
    border-radius: 12px;
    border-left: 5px solid #10b981;
}

.warning-box {
    background: #fff7ed;
    color: #9a3412;
    padding: 14px;
    border-radius: 12px;
    border-left: 5px solid #f97316;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Model Architecture
# -----------------------------
def build_unet(input_shape=(256, 256, 3)):
    inputs = layers.Input(shape=input_shape)

    c1 = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    b1 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(p3)
    b1 = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(b1)

    u1 = layers.UpSampling2D((2, 2))(b1)
    u1 = layers.concatenate([u1, c3])
    c4 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(u1)
    c4 = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(c4)

    u2 = layers.UpSampling2D((2, 2))(c4)
    u2 = layers.concatenate([u2, c2])
    c5 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(u2)
    c5 = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(c5)

    u3 = layers.UpSampling2D((2, 2))(c5)
    u3 = layers.concatenate([u3, c1])
    c6 = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(u3)
    c6 = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(c6)

    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(c6)

    return models.Model(inputs=[inputs], outputs=[outputs])


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = build_unet()
    weights_path = "segmentation.weights.h5"

    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        return model, True
    return model, False


model, weights_loaded = load_model()


# -----------------------------
# Helper Functions
# -----------------------------
def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((256, 256))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch, img_resized


def create_overlay(original_img, mask, opacity=0.45):
    original = np.array(original_img).astype(np.uint8)
    mask_uint8 = (mask * 255).astype(np.uint8)

    color_mask = np.zeros_like(original)
    color_mask[:, :, 0] = mask_uint8

    overlay = cv2.addWeighted(original, 1 - opacity, color_mask, opacity, 0)
    return overlay


def image_to_bytes(image_array):
    img = Image.fromarray(image_array.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("🧠 GI Segmentation AI")

    st.markdown("### Model Settings")
    threshold = st.slider("Mask Threshold", 0.10, 0.90, 0.50, 0.05)
    opacity = st.slider("Overlay Opacity", 0.10, 0.90, 0.45, 0.05)

    st.markdown("### Display Settings")
    image_width = st.slider("Image Width", 250, 650, 400, 50)

    st.markdown("---")

    if weights_loaded:
        st.success("Model weights loaded successfully")
    else:
        st.warning("Weights file not found. Using untrained model.")

    st.markdown("""
    ### About
    This app uses a U-Net deep learning model to segment gastrointestinal structures from medical images.
    """)


# -----------------------------
# Main UI
# -----------------------------
st.markdown('<div class="main-title">Gastrointestinal Image Segmentation Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a medical image and generate segmentation masks, overlays, metrics, and downloadable results.</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload GI Image",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG or PNG image for segmentation."
)

if uploaded_file is None:
    st.markdown("""
    <div class="card">
        <h3>Upload an image to begin</h3>
        <p>This dashboard will show:</p>
        <ul>
            <li>Original image</li>
            <li>Predicted segmentation mask</li>
            <li>Overlay visualization</li>
            <li>Segmentation metrics</li>
            <li>Downloadable results</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

else:
    input_image, original_pil = preprocess_image(uploaded_file)

    with st.spinner("Running U-Net segmentation..."):
        prediction = model.predict(input_image, verbose=0)[0, :, :, 0]

    binary_mask = prediction > threshold
    mask_image = (binary_mask.astype(np.uint8) * 255)
    overlay_image = create_overlay(original_pil, binary_mask.astype(np.uint8), opacity)

    segmented_pixels = int(np.sum(binary_mask))
    total_pixels = binary_mask.size
    coverage_percent = (segmented_pixels / total_pixels) * 100
    avg_confidence = float(np.mean(prediction) * 100)
    max_confidence = float(np.max(prediction) * 100)

    st.markdown('<div class="success-box">Segmentation completed successfully.</div>', unsafe_allow_html=True)

    st.markdown("## Segmentation Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Original Image")
        st.image(original_pil, width=image_width)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Predicted Mask")
        st.image(mask_image, width=image_width, clamp=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Overlay Image")
        st.image(overlay_image, width=image_width)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("## Model Output Metrics")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{coverage_percent:.2f}%</div>
            <div class="metric-label">Mask Coverage</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{segmented_pixels}</div>
            <div class="metric-label">Segmented Pixels</div>
        </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_confidence:.2f}%</div>
            <div class="metric-label">Average Prediction Score</div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{max_confidence:.2f}%</div>
            <div class="metric-label">Max Prediction Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## Analyst Interpretation")

    if coverage_percent < 1:
        interpretation = "Very small segmented region detected. This may indicate limited visible GI structure or a weak prediction."
    elif coverage_percent < 10:
        interpretation = "A small but noticeable segmented region was detected. Review the overlay carefully to confirm whether the mask aligns with the target structure."
    elif coverage_percent < 35:
        interpretation = "A moderate segmented region was detected. The prediction appears to identify a meaningful structure within the image."
    else:
        interpretation = "A large segmented region was detected. Check whether the model is over-segmenting the image."

    st.markdown(f"""
    <div class="card">
        <h3>Summary</h3>
        <p>{interpretation}</p>
        <p><b>Note:</b> This tool is for academic/demo use only and should not be used for clinical diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("## Download Results")

    d1, d2 = st.columns(2)

    with d1:
        st.download_button(
            label="Download Mask Image",
            data=image_to_bytes(np.stack([mask_image] * 3, axis=-1)),
            file_name="predicted_mask.png",
            mime="image/png"
        )

    with d2:
        st.download_button(
            label="Download Overlay Image",
            data=image_to_bytes(overlay_image),
            file_name="overlay_result.png",
            mime="image/png"
        )