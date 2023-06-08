import cv2
import numpy as np

# Create a VideoCapture object to read from the camera
cap = cv2.VideoCapture(0)

# Loop over frames from the video stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper red color ranges in HSV
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Threshold the HSV image to get only the red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine the masks
    mask = cv2.bitwise_or(mask1, mask2)

    # Apply morphological operations to remove noise from the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Changed to MORPH_CLOSE

    # Find contours of the red regions
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # Changed to RETR_LIST

    # Draw contours on the frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # Display the original frame with contours
    cv2.imshow('Frame', frame)

    # Wait
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key is pressed, break from the loop
    if key == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
