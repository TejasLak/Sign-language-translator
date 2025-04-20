import streamlit as st
import cv2
from utils.preprocessing import process_frame
from tensorflow.keras.models import load_model
import numpy as np
import json

st.title("Sign Language Recognition")
language = st.selectbox("Select Sign Language", ['TEST', 'BSL', 'ISL'])

model_path = f"models/{language.lower()}_model.h5"
labels_path = f"models/{language.lower()}_class_indices.json"

# Load model
model = load_model(model_path)

# Load class labels
with open(labels_path, "r") as f:
    class_indices = json.load(f)

# Reverse the dictionary to map indices to labels
idx_to_label = {v: k for k, v in class_indices.items()}

cap = cv2.VideoCapture(0)
stframe = st.empty()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    processed = process_frame(frame)  # Apply Mediapipe hand detection and crop
    prediction = model.predict(np.expand_dims(processed, axis=0))
    label_index = int(np.argmax(prediction))
    label = idx_to_label.get(label_index, "Unknown")

    stframe.image(frame, channels="BGR")
    st.write("Prediction:", label)

cap.release()
cv2.destroyAllWindows()
