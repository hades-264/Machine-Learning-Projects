import cv2
import numpy as np

cap = cv2.VideoCapture(0)
back = cv2.imread('./image.jpg')
while cap.isOpened():
    # take each frame
    ret, frame = cap.read()

    if ret:
        # convert RBG to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # cv2.imshow("hsv", hsv)

        # RBG Red Color
        red = np.uint8([[[0, 0, 255]]])
        # HSV Red Color
        hsv_red = cv2.cvtColor(red, cv2.COLOR_BGR2HSV)

        # upper and lower boundary for red color for highlighting
        l_red = np.array([0, 100, 100])
        u_red = np.array([10, 255, 255])

        mask = cv2.inRange(hsv, l_red, u_red)
        # cv2.imshow("mask", mask)

        # all things red
        part1 = cv2.bitwise_and(back, back, mask=mask)
        # cv2.imshow("part1", part1)

        mask = cv2.bitwise_not(mask)

        # all things not red
        part2 = cv2.bitwise_and(frame, frame, mask=mask)
        # cv2.imshow("mask", part2)

        # this will cloak
        cv2.imshow("cloak",part1+part2)

        k = cv2.waitKey(30) & 0xff
        # if esc key is pressed break the loop and exit
        if k == 27:
            break
cap.release()
