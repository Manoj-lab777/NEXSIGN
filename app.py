import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap');
    .nexsign-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 50px;
        font-weight: 700;
        color: #4A90E2;
        letter-spacing: 2px;
    }
    </style>
    <div class="nexsign-title">NexSign</div>
""", unsafe_allow_html=True)

st.subheader("ISL to English Translator")

actions = np.array([
    'hello', 'thank_you', 'please', 'sorry', 'yes',
    'no', 'help', 'stop', 'go', 'come',
    'eat', 'drink', 'water', 'name', 'good',
    'bad', 'happy', 'sad', 'love', 'friend'
])

model = load_model('models/isl_model.h5')

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])

run = st.checkbox('Start Camera')
FRAME_WINDOW = st.image([])
prediction_text = st.empty()

sequence = []

if run:
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.write("Camera not found")
                break

            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predicted_word = actions[np.argmax(res)]
                confidence = np.max(res) * 100
                prediction_text.markdown(f"### Sign detected: **{predicted_word}** ({confidence:.1f}% confidence)")

            FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    cap.release()
else:
    st.write("Check the box above to start the camera")  