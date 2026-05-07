import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 
                     'ml')
                         )
                     )
from model import SimpleUNet

st.set_page_config(
    page_title="SVAMITVA AI Feature Extraction",
    page_icon="🛰️",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.title {
    font-size: 42px;
    font-weight: 800;
    color: #0B1F3A;
}
.subtitle {
    font-size: 18px;
    color: #4B5563;
}
.card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0 4px 14px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.metric-box {
    background: #eef4ff;
    padding: 18px;
    border-radius: 14px;
    text-align: center;
    font-weight: bold;
}
.legend {
    font-size: 16px;
    line-height: 2;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🛰️ SVAMITVA AI Feature Extraction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-based detection of building footprints, roads and water bodies from drone orthophotos.</div>',
    unsafe_allow_html=True
)

st.sidebar.title("📌 Project Panel")
st.sidebar.info("""
**Project:** Drone Orthophoto Feature Extraction  
**Model:** Simplified U-Net  
**Classes:** Building, Road, Water Body  
**Tech:** PyTorch + OpenCV + Streamlit
""")

st.sidebar.markdown("### 🎨 Legend")
st.sidebar.markdown("""
🟢 Building Footprint  
🟡 Road  
🔵 Water Body  
""")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = SimpleUNet().to(device)

    model_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "saved_models", "model.pth")
                                                      )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

model = load_model()

st.markdown("---")

left, right = st.columns([1, 2])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📤 Upload Drone Image")
    uploaded_file = st.file_uploader(
        "Choose an orthophoto image",
        type=["jpg", "jpeg", "png"]
    )
    st.write("Upload a drone image to detect land features.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📊 Model Information")
    st.write("Input Size: 128 × 128")
    st.write("Output: Segmented feature map")
    st.write("Status: Demo-ready prototype")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    if uploaded_file is None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("👋 Welcome")
        st.write("Upload an image from the left panel to start feature extraction.")
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Drone_icon.svg/512px-Drone_icon.svg.png", width=180)
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        image_resized = cv2.resize(image_np, (128, 128))

        input_image = image_resized / 255.0
        input_tensor = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

        with st.spinner("AI model is detecting features..."):
            with torch.no_grad():
                output = model(input_tensor)
                prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()

        colored_mask = np.zeros((128, 128, 3), dtype=np.uint8)
        colored_mask[prediction == 1] = [0, 255, 0]
        colored_mask[prediction == 2] = [255, 255, 0]
        colored_mask[prediction == 3] = [0, 0, 255]

        overlay = cv2.addWeighted(image_resized, 0.6, colored_mask, 0.4, 0)

        st.success("Feature extraction completed successfully!")

        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🖼️ Original Image")
            st.image(image_resized, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("✅ Detected Features")
            st.image(overlay, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("### 📌 Detection Summary")
        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown('<div class="metric-box">🟢 Building<br>Detected</div>', unsafe_allow_html=True)
        with m2:
            st.markdown('<div class="metric-box">🟡 Road<br>Detected</div>', unsafe_allow_html=True)
        with m3:
            st.markdown('<div class="metric-box">🔵 Water Body<br>Detected</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("Developed for AI-based feature extraction from drone orthophotos under SVAMITVA context.")