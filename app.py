import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime

st.title("Face Recognition Demo")

# Load the face recognizer and face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Directory where the face data is stored
DATA_DIR = "face_data"

# Train the recognizer from uploaded images
def train_model():
    images = []
    labels = []
    label_dict = {}
    target_size = (200, 200)

    for i, file in enumerate(os.listdir(DATA_DIR)):
        image_path = os.path.join(DATA_DIR, file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, target_size)
        label = file.split('_')[0]

        if label not in label_dict:
            label_dict[label] = len(label_dict)

        images.append(image)
        labels.append(label_dict[label])

    recognizer.train(images, np.array(labels))
    return label_dict

# Simulated live recognition using webcam snapshot
image_data = st.camera_input("Take a picture")
if image_data:
    label_dict = train_model()
    file_bytes = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (200, 200))
        label, confidence = recognizer.predict(face)

        person_name = "Unknown"
        for name, id in label_dict.items():
            if id == label:
                person_name = name
                break

        if confidence < 50:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            st.write(f"✅ Recognized: {person_name} at {now}")
        else:
            st.write("❌ Unidentified face")

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, person_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    st.image(img, channels="BGR", caption="Recognition Result")
