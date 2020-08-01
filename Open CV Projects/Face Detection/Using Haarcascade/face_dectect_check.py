import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

data = np.load("face_data.npy")

X = data[:, 1:].astype(int)
y = data[:, 0]

# plot data on the graph
model = KNeighborsClassifier()
model.fit(X, y)

cap = cv2.VideoCapture(0)

# you can simply use the packaged path to the installed cascades - cv2.data.haarcascades:
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # will read data from cap until manually stoped using some keyStroke
    ret, frame = cap.read()

    if ret:
        faces = detector.detectMultiScale(frame)

        for face in faces:
            x, y, w, h = face
            faceCut = frame[y: y + h, x: x + w]

            fix = cv2.resize(faceCut, (100, 100))
            gray = cv2.cvtColor(fix, cv2.COLOR_BGR2GRAY)

            out = model.predict([gray.flatten()])

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.putText(frame, str(out[0]), (x, y - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)

            print(out)

            cv2.imshow("My Face", gray)

        cv2.imshow("My Screen", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
