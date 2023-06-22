import cv2

# Creating a VideoCapture object to read from the camera
cap = cv2.VideoCapture(0)

# List to store coordinates of all the choicen points
coords = []

# Callback function that prints the coordinates for the given point
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Adds the coordinates 
        coords.append((x, y))
        print(f"Coordinates at click: {x}, {y}")

# Registers callback function
cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', mouse_callback)

# Loop over frames
while True:
    # Read a frame 
    ret, frame = cap.read()

    # If mouse is clicked, make points at the click locations
    for coord in coords:
        cv2.circle(frame, coord, 5, (0, 0, 255), -1)

    # Display the frame
    cv2.imshow('Frame', frame)

  
    key = cv2.waitKey(1) & 0xFF

    # If  q key is pressed, break from the loop
    if key == ord('q'):
        break

#  Print all stored coordinates
print(f"hardcoded_coordinates = {coords}")

# Close everything
cap.release()
cv2.destroyAllWindows()
