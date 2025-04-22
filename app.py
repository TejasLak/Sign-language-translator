import streamlit as st
import cv2
import numpy as np
import json
import time
from tensorflow.keras.models import load_model
from utils.preprocessing import process_frame

st.set_page_config(layout="wide")
st.title("Sign Language Recognition - Sentence Builder")

# Select language
language = st.selectbox("Select Sign Language", ['TEST', 'BSL', 'ISL'])

# Load model and labels
model_paths = {
    "TEST": "models/test_mediapipe_model.h5",
    "BSL": "models/bsl_model.h5",
    "ISL": "models/isl_model.h5"
}
labels_paths = {
    "TEST": "models/test_class_indices_mediapipe.json",
    "BSL": "models/bsl_class_indices.json",
    "ISL": "models/isl_class_indices.json"
}

model = load_model(model_paths[language])
with open(labels_paths[language], "r") as f:
    label_map = json.load(f)
    label_map = {v: k for k, v in label_map.items()}

# Streamlit UI containers
col1, col2 = st.columns([1, 2])
stframe = col1.empty()
sentence_display = col2.empty()

# Video capture
cap = cv2.VideoCapture(0)

# Sentence variables
sentence = ""
prev_letter = ""
last_prediction_time = 0
time_threshold = 1.5  # seconds
confidence_threshold = 0.8

# Reset button
if st.button("Reset Sentence"):
    sentence = ""
    prev_letter = ""
    last_prediction_time = 0

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process with Mediapipe
    processed = process_frame(frame)
    prediction = model.predict(np.expand_dims(processed, axis=0))[0]
    predicted_letter = label_map[np.argmax(prediction)]
    confidence = np.max(prediction)
    current_time = time.time()

    # Add to sentence if confident and not repeated too quickly
    if confidence > confidence_threshold and predicted_letter != prev_letter:
        if current_time - last_prediction_time > time_threshold:
            sentence += predicted_letter
            prev_letter = predicted_letter
            last_prediction_time = current_time

    # Show webcam and sentence
    stframe.image(frame, channels="BGR")
    sentence_display.markdown(f"## ✍️ Sentence: `{sentence}`")

cap.release()
cv2.destroyAllWindows()
