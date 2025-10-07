import streamlit as st
import cv2
import numpy as np
from datetime import datetime

st.title("Face Detection Demo")

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Simulated live capture using webcam snapshot
image_data = st.camera_input("Take a picture")
if image_data:
    file_bytes = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, "Unidentified", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    st.image(img, channels="BGR", caption="Detected Faces")
    st.write(f"Detected {len(faces)} face(s) at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
else:
    st.info("Please take a picture to begin face detection.")

