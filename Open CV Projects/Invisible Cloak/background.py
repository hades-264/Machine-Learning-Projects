import cv2

# start the camers
cap = cv2.VideoCapture(0)

# function used to check if camera is on
while cap.isOpened():
    # back is the background
    ret, back = cap.read()
    if ret:
        cv2.imshow("img",back)

        k = cv2.waitKey(30) & 0xff
        # if esc key is pressed save the background, break the loop and exit
        if k == 27:
            cv2.imwrite('image.jpg',back)
            break
cap.release()







