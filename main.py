import face_recognition
import cv2
import numpy as np
import pyautogui
import time

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a second sample picture and learn how to recognize it.
# Move a photo of your face to the assets folder and change the name below
my_image = face_recognition.load_image_file("/assets/<your_photo>")
my_face_encoding = face_recognition.face_encodings(face_image=my_image)
# Create arrays of known face encodings and their names
known_face_encodings = [my_face_encoding]
known_face_names = ["Me"]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations
        )

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                known_face_encodings[0], face_encoding
            )
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(
                known_face_encodings[0], face_encoding
            )
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

    if "Me" in face_names:
        pyautogui.press("enter")
        # wait for the password screen to load
        time.sleep(2)
        pyautogui.write("your password", interval=0.25)
        pyautogui.sleep(1)
        pyautogui.press("enter")

        break

# Release handle to the webcam
video_capture.release()
