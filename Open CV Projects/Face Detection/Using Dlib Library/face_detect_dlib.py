import cv2
import dlib

# start the video camera for capturing frames
cap = cv2.VideoCapture(0)

# frontal face detector
detector = dlib.get_frontal_face_detector()

# for landmarkiqng
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")

while True:
    # will read data from cap until manually stoped using some keyStroke
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if ret:
        faces = detector(frame)

        for face in faces:
            landmarks = predictor(gray, face)
            # nose = landmarks.parts()[27]
            for point in landmarks.parts():
                cv2.circle(frame, (point.x, point.y), 1, (255, 0, 0), 3)

        if ret:  # ie if camera is working then update the frame on the screen
            cv2.imshow("My Screen", frame)

    key = cv2.waitKey(1)

    # if key 'q' is pressed then stop the camera
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
