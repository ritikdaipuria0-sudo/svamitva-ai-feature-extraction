import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR / "ml"))

from model import SimpleUNet
import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import sys
import os

sys.path.append("../ml")

from model import SimpleUNet

st.set_page_config(page_title="SVAMITVA AI Feature Extraction", layout="wide")

st.title("AI Model for Feature Extraction from Drone Orthophotos")
st.write("Upload a drone orthophoto and detect Building, Road, and Water Body features.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = SimpleUNet().to(device)
    MODEL_PATH = ROOT_DIR / "saved_models" / "model.pth"
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model

model = load_model()

uploaded_file = st.file_uploader("Upload Drone Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_resized = cv2.resize(image_np, (128, 128))

    input_image = image_resized / 255.0
    input_tensor = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).squeeze().cpu().numpy()

    colored_mask = np.zeros((128, 128, 3), dtype=np.uint8)
    colored_mask[prediction == 1] = [0, 255, 0]
    colored_mask[prediction == 2] = [255, 255, 0]
    colored_mask[prediction == 3] = [0, 0, 255]

    overlay = cv2.addWeighted(image_resized, 0.6, colored_mask, 0.4, 0)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image_resized, channels="RGB")

    with col2:
        st.subheader("Detected Features")
        st.image(overlay, channels="RGB")

    st.markdown("""
    ### Legend
    - 🟢 Building Footprint
    - 🟡 Road
    - 🔵 Water Body
    """)