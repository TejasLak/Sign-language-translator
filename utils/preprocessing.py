import numpy as np
import cv2
import mediapipe as mp

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

def process_frame(frame):
    """
    Uses Mediapipe to detect and crop the hand, then resizes and normalizes the frame.
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        h, w, _ = frame.shape
        hand_landmarks = results.multi_hand_landmarks[0]

        # Get bounding box coordinates
        x_coords = [lm.x for lm in hand_landmarks.landmark]
        y_coords = [lm.y for lm in hand_landmarks.landmark]

        x_min = int(min(x_coords) * w) - 20
        x_max = int(max(x_coords) * w) + 20
        y_min = int(min(y_coords) * h) - 20
        y_max = int(max(y_coords) * h) + 20

        # Clamp values to frame size
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        cropped = frame[y_min:y_max, x_min:x_max]

        if cropped.size == 0:
            cropped = frame  # fallback to original if bad crop
    else:
        cropped = frame  # fallback if no hand detected

    resized = cv2.resize(cropped, (64, 64))
    normalized = resized / 255.0
    return np.array(normalized, dtype=np.float32)
