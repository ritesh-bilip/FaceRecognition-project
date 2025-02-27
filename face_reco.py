import cv2
import time
import imutils
import os
import face_recognition

# Initialize the camera
cam = cv2.VideoCapture(0)
time.sleep(1)

known_face_encodings = []
known_face_names = []

# Define a dictionary to map filenames to specific names
name_mapping = {
    "person1.jpg": "Ritesh",
    "person2.jpg": "Bikrom",
    "person3.jpg": "Rocky",
    "person4.jpg": "Subro"
}

# Load images and encode faces
folder_path = r'C:\Users\dasri\opencv\images'  # Verify this path exists
for filename in os.listdir(folder_path):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        img = face_recognition.load_image_file(os.path.join(folder_path, filename))
        encoding = face_recognition.face_encodings(img)
        if encoding:  # Check if encoding is found
            known_face_encodings.append(encoding[0])
            known_face_names.append(name_mapping.get(filename, "Unknown"))

while True:
    ret, img = cam.read()
    text = "Unknown"

    img = imutils.resize(img, width=640)
    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        text = name
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(img, text, (left + 6, top - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Camera Feed", img)

    key = cv2.waitKey(10)
    if key == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
