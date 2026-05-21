import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os

st.set_page_config(page_title="Rice Disease Detector", layout="centered")
st.title("🌾 Rice Disease Detection (YOLOv8)")

WEIGHTS_PATH = 'best.pt'

@st.cache_resource
def load_model():
    if os.path.exists(WEIGHTS_PATH):
        return YOLO(WEIGHTS_PATH)
    return None

model = load_model()

if model is None:
    st.error(f"Model weights not found. Ensure 'best.pt' is in the same folder.")
else:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        if st.button('Run Detection'):
            results = model.predict(source=image, conf=0.25)
            res_plotted = results[0].plot()[:, :, ::-1]
            st.image(res_plotted, caption='Detection Results', use_container_width=True)
