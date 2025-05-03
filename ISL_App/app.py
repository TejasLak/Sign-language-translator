import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import os
from preprocessing import process_frame

# Load the trained model
MODEL_PATH = 'ISL_App/models/isl_mediapipe_model.h5'  # Replace with your model path
LABEL_PATH = 'ISL_App/models/isl_class_indices_mediapipe.json'  # Replace with your label path

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_PATH, 'r') as f:
    class_indices = json.load(f)
    class_names = {v: k for k, v in class_indices.items()}

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Streamlit app
st.title("Indian Sign Language Recognition")
st.sidebar.title("Controls")

# Initialize session state
if 'sentence' not in st.session_state:
    st.session_state.sentence = ''

# Buttons
if st.sidebar.button("Add Space"):
    st.session_state.sentence += ' '

if st.sidebar.button("Clear Sentence"):
    st.session_state.sentence = ''

# Display the current sentence
st.markdown(f"### Sentence: {st.session_state.sentence}")

# Start video capture
cap = cv2.VideoCapture(0)
FRAME_WINDOW = st.image([])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning("Failed to capture video")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        # Initialize list for 42 landmarks (21 per hand)
        landmarks = np.zeros((42, 3))
        for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            for i, lm in enumerate(hand_landmarks.landmark):
                if idx == 0:
                    landmarks[i] = [lm.x, lm.y, lm.z]
                elif idx == 1:
                    landmarks[i + 21] = [lm.x, lm.y, lm.z]

        processed_frame = process_frame(frame)
        if processed_frame is not None:
           input_data = np.expand_dims(processed_frame, axis=0)  # Shape becomes (1, 128, 128, 3)
           prediction = model.predict(input_data)
        else:
           # Handle the case where no hands are detected
           pass

        predicted_class = np.argmax(prediction)
        predicted_label = class_names.get(predicted_class, '')

        # Display prediction
        cv2.putText(frame, f'Prediction: {predicted_label}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Add to sentence if confidence is high
        confidence = prediction[0][predicted_class]
        if confidence > 0.8:
            st.session_state.sentence += predicted_label + ' '

    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
