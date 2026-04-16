import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

st.set_page_config(page_title="Road Scene Detection System", layout="centered")

st.title("Road Scene Detection System")
st.write(
    "Upload a road image and run real-time object detection using YOLOv8."
)

model = YOLO("yolov8n.pt")

uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    results = model(image)
    annotated = results[0].plot()

    st.image(
        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        caption="Detection Result",
        use_container_width=True,
    )
