import cv2


cap = cv2.VideoCapture(0)

# List to store coordinates of all the choicen points
coords = []

#  Function to print the coordinates for the given click
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Adds the coordinates 
        coords.append((x, y))
        print(f"Coordinates at click: {x}, {y}")


cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', mouse_callback)


while True:
 
    ret, frame = cap.read()

    # If mouse is clicked, make points at the click locations
    for coord in coords:
        cv2.circle(frame, coord, 5, (0, 0, 255), -1)


    cv2.imshow('Frame', frame)

  
    key = cv2.waitKey(1) & 0xFF


    if key == ord('q'):
        break


print(f"hardcoded_coordinates = {coords}")


cap.release()
cv2.destroyAllWindows()
