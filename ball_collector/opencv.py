import cv2
import numpy as np

# Create a VideoCapture object to read from the camera
cap = cv2.VideoCapture(1)

# Loop over frames from the video stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of red color in HSV
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    #creating the Red color
    red_mask = mask1 + mask2

    # Define the range of white color in HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 20, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Define the range of green color in HSV
    lower_green = np.array([36, 25, 25])
    upper_green = np.array([86, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Define the range of orange color in HSV
    lower_orange = np.array([10, 100, 20])
    upper_orange = np.array([25, 255, 255])
    orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Combine the red, white, green, and orange masks
    mask = red_mask + white_mask + green_mask + orange_mask

    # Apply the mask to the original frame
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the original frame and the result with the color masks applied
    cv2.imshow('Original', frame)
    cv2.imshow('Result', result)

    # Wait
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key is pressed, break from the loop
    if key == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()