import streamlit as st
import cv2
import numpy as np
import face_recognition

st.title("Face Recognition Demo")

# Simulated live capture using webcam snapshot
image_data = st.camera_input("Live Face Recognition")
if image_data:
    file_bytes = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Convert to RGB for face_recognition
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Load known face encodings (empty for now)
    known_face_encodings = []
    known_face_names = []

    # Detect faces and compute encodings
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unidentified"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Draw box and label
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    st.image(img, channels="BGR", caption="Recognition Result")

