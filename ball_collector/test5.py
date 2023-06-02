import cv2
import numpy as np
import math

# Function to detect the green and pink rectangles (front and back of the robot)
def detect_robot(frame):
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the HSV values for the green and pink rectangles
    hsv_values = {
        'green': {'hmin': 70, 'smin': 38, 'vmin': 0, 'hmax': 92, 'smax': 255, 'vmax': 255},
        'pink': {'hmin': 140, 'smin': 100, 'vmin': 100, 'hmax': 180, 'smax': 255, 'vmax': 255}
    }

    # Threshold the HSV image to get only green and pink colors
    mask_green = get_mask(hsv, hsv_values['green'])
    mask_pink = get_mask(hsv, hsv_values['pink'])

    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    mask_pink = cv2.morphologyEx(mask_pink, cv2.MORPH_OPEN, kernel)

    # Find contours of green and pink regions
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_pink, _ = cv2.findContours(mask_pink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the green robot (front)
    green_robot = find_largest_contour(contours_green)

    # Find the pink robot (back)
    pink_robot = find_largest_contour(contours_pink)

    return green_robot, pink_robot


# Function to create a binary mask for a given HSV range
def get_mask(image, hsv_range):
    lower_range = np.array([hsv_range['hmin'], hsv_range['smin'], hsv_range['vmin']])
    upper_range = np.array([hsv_range['hmax'], hsv_range['smax'], hsv_range['vmax']])
    mask = cv2.inRange(image, lower_range, upper_range)
    return mask


# Function to find the largest contour from a list of contours
def find_largest_contour(contours):
    largest_contour = None
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour
    return largest_contour


# Function to calculate the angle of the robot
def calculate_angle(green_robot, pink_robot):
    if green_robot is not None and pink_robot is not None:
        green_center = np.mean(green_robot, axis=0)[0]
        pink_center = np.mean(pink_robot, axis=0)[0]
        angle = math.atan2(pink_center[1] - green_center[1], pink_center[0] - green_center[0])
        angle_degrees = math.degrees(angle)
        return angle_degrees
    else:
        return None


# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from the webcam
    ret, frame = cap.read()

    # Detect the green and pink rectangles (front and back of the robot)
    green_robot, pink_robot = detect_robot(frame)

    # Calculate the angle of the robot
    angle = calculate_angle(green_robot, pink_robot)

    # Update the line position based on the detected robot positions
    if green_robot is not None and pink_robot is not None:
        green_center = np.mean(green_robot, axis=0)[0]
        pink_center = np.mean(pink_robot, axis=0)[0]

        # Calculate the line start and end positions
        line_start = (int(green_center[0]), int(green_center[1]))
        line_end = (int(pink_center[0]), int(pink_center[1]))

        # Draw the line
        line_color = (0, 255, 0)  # Green color
        line_thickness = 2
        cv2.line(frame, line_start, line_end, line_color, line_thickness)

        # Calculate the angle text position
        text_position = (int((line_start[0] + line_end[0]) / 2), int((line_start[1] + line_end[1]) / 2))
        text = f"Angle: {angle:.2f} degrees"

        # Draw the angle text
        cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, line_color, 2)

    # Display the frame
    cv2.imshow('Robot Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
