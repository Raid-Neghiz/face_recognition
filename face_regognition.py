
import cv2
import streamlit as st

def detect_face():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Unable to access the camera. Please check your device.")
        return

    frame_placeholder = st.empty()
    stop_button = st.button("Stop Detection")

    while True:
        ret, frame = cap.read()

        if not ret:
            st.error('Failed to capture frame. Exiting...')
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in face:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_placeholder.image(rgb_frame, channels='RGB', use_container_width=True)

        if stop_button:
            break

    cap.release()

def main():
    st.title("Real-Time Face Detection")

    st.write("Welcome! This app uses your webcam to detect faces in real-time.")
    st.write("Click the **Start Detection** button below to begin.")

    st.sidebar.header("Detection Settings")
    scale_factor = st.sidebar.slider("Scale Factor", 1.1, 2.0, 1.3, 0.1)
    min_neighbors = st.sidebar.slider("Min Neighbors", 3, 10, 5)

    if st.button('Start Detection'):
        st.write("Click the **Stop Detection** button to end the session.")
        detect_face()

if __name__ == '__main__':
    main()
