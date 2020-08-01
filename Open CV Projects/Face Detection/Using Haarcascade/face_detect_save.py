import cv2
import numpy as np
import os

# start the video camera for capturing frames
cap = cv2.VideoCapture(0)

# input name for saving the picture
name = input("Enter Your Name : ")

frames = []
outputs = []

# you can simply use the packaged path to the installed cascades - cv2.data.haarcascades:
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # will read data from cap until manually stoped using some keyStroke
    ret, frame = cap.read()

    if ret:
        faces = detector.detectMultiScale(frame)

        for face in faces:
            # x axis y axis height from top and from ending from x axis to end of face
            x, y, w, h = face
            faceCut = frame[y: y + h, x: x + w]

            # this will hvae a fixed frame for face
            fix = cv2.resize(faceCut, (200, 200))
            # convert fixed frame to Black and White
            gray = cv2.cvtColor(fix, cv2.COLOR_BGR2GRAY)

        # if ret == True ie if camera is working then update the frame on the screen
        cv2.imshow("My Screen", frame)
        cv2.imshow("My Face", gray)

    key = cv2.waitKey(1)

    # if key 'c' is pressed then capture image and save it
    # if key == ord("c"):
    #      cv2.imwrite(name + ".jpg", frame)

    # if key 'c' is pressed then capture image and append it
    if key == ord("c"):
        frames.append(gray.flatten())
        outputs.append([name])

    # if key 'q' is pressed then stop the camera
    if key == ord("q"):
        break

# x->photo y->names
x = np.array(frames)
y = np.array(outputs)

# hstack() function is used to stack the sequence of input arrays horizontally
#  (i.e. column wise) to make a single array.
data = np.hstack([y, x])
face_name = "face_data.npy"

# if file exits add data to it rather than making a new file
if os.path.exists(face_name):
    old = np.load(face_name)
    data = np.vstack([old, data])

# save data ie frame and output
np.save(face_name, data)

# this will have(x->number of images,y->number of features)
# print(data.shape)

cap.release()
cv2.destroyAllWindows()
