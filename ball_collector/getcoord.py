import cv2

# Create a VideoCapture object to read from the camera
cap = cv2.VideoCapture(0)

# List to store coordinates
coords = []

# Callback function for mouse events
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add the coordinates to the list
        coords.append((x, y))
        print(f"Coordinates at click: {x}, {y}")

# Register the mouse callback function
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', mouse_callback)

# Loop over frames from the video stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # If there were any clicks, draw circles at the click locations
    for coord in coords:
        cv2.circle(frame, coord, 5, (0, 0, 255), -1)

    # Display the frame
    cv2.imshow('Frame', frame)

    # Wait
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key is pressed, break from the loop
    if key == ord('q'):
        break

# After breaking the loop, print all stored coordinates
print(f"hardcoded_coordinates = {coords}")

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
