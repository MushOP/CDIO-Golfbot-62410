import cv2
import numpy as np

# Define the HSV structure that we need 
hsv_values = {'hmin': 140, 'smin': 100, 'vmin': 100, 'hmax': 180, 'smax': 255, 'vmax': 255}


cap = cv2.VideoCapture(0)

# Callback function for mouse click
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Get the HSV value at the clicked pixel
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_value = hsv[y, x]
        print("HSV value at ({}, {}): {}".format(x, y, hsv_value))

        #  The HSV value range
        hmin, smin, vmin = np.maximum(hsv_value - np.array([40, 155, 155]), 0)
        hmax, smax, vmax = np.minimum(hsv_value + np.array([40, 100, 100]), [180, 255, 255])

        # Print the HSV value 
        hsv_values = {'hmin': hmin, 'smin': smin, 'vmin': vmin, 'hmax': hmax, 'smax': smax, 'vmax': vmax}
        print("HSV value range: {}".format(hsv_values))


cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', mouse_callback)


while True:
    # Reads a frame 
    ret, frame = cap.read()

    # Displays the frame
    cv2.imshow('Frame', frame)

    
    key = cv2.waitKey(1) & 0xFF

    # If q is pressed, break from the loop
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
