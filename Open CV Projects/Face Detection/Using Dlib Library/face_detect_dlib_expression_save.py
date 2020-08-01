import cv2
import dlib
import numpy as np
import os

# start the video camera for capturing frames
cap = cv2.VideoCapture(0)

# frontal face detector
detector = dlib.get_frontal_face_detector()

# for landmarkiqng
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

mood = input("Enter your mood : ")

frames = []
outputs = []

while True:
    # will read data from cap until manually stoped using some keyStroke
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ret:
        faces = detector(frame)

        for face in faces:
            landmarks = predictor(gray, face)
            expression = np.array([[point.x - face.left(), point.y - face.top()]
                               for point in landmarks.parts()[17:]])
    
    if ret:
        cv2.imshow("My Screen", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break
    elif key == ord("c"):
        frames.append(expression.flatten())
        outputs.append([mood])


X = np.array(frames)
y = np.array(outputs)

data = np.hstack([y, X])

f_name = "face_data.npy"

if os.path.exists(f_name):
    old = np.load(f_name)
    data = np.vstack([old, data])

np.save(f_name, data)

cap.release()
cv2.destroyAllWindows()
