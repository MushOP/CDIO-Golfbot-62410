import cv2
import numpy as np
import heapq
import math

# Create a VideoCapture object to read from the camera
cap = cv2.VideoCapture(0)
MAX_DISTANCE = 1000000

# Function to calculate the Euclidean distance between two points
def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Function to find the shortest path using Dijkstra's algorithm
def dijkstra(graph, start):
    distances = {node: math.inf for node in graph}
    distances[start] = 0
    queue = [(0, start)]

    while queue:
        current_distance, current_node = heapq.heappop(queue)

        if current_distance > distances[current_node]:
            continue

        if current_node not in graph:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(queue, (distance, neighbor))

    return distances

# Function to detect the green rectangle (robot)
def detect_robot(frame):
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the HSV values for the green rectangle
    hsv_values = {'hmin': 70, 'smin': 38, 'vmin': 0, 'hmax': 92, 'smax': 255, 'vmax': 255}
    lower_green = np.array([hsv_values['hmin'], hsv_values['smin'], hsv_values['vmin']])
    upper_green = np.array([hsv_values['hmax'], hsv_values['smax'], hsv_values['vmax']])

    # Threshold the HSV image to get only green colors
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

    # Find contours of green regions
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the robot (green rectangle)
    robot = None
    largest_contour = None
    max_area = 0
    for contour in contours_green:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            largest_contour = contour

    if largest_contour is not None:
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 3)
        center = np.mean(box, axis=0)
        robotH = math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
        robotW = math.sqrt((box[1][0] - box[2][0]) ** 2 + (box[1][1] - box[2][1]) ** 2)
        robot = (int(center[0]), int(center[1]), int(robotH), int(robotW))

    return robot

def detect_white_balls(frame):
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of white color in HSV
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 20, 255])

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours of white regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area and shape
    min_area = 100  # Minimum area threshold for contours
    max_area = 2000  # Maximum area threshold for contours
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            valid_contours.append(contour)

    # Draw bounding boxes around valid contours
    for contour in valid_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Extract valid contour coordinates
    coordinates = [(cv2.boundingRect(contour)[:2]) for contour in valid_contours]

    # Create a graph with valid contour coordinates as vertices
    graph = {coord: {} for coord in coordinates}

    # Connect each vertex in the graph with its neighboring vertices
    for v1 in graph:
        for v2 in graph:
            if v1 != v2:
                distance = np.sqrt((v2[0] - v1[0]) ** 2 + (v2[1] - v1[1]) ** 2)
                if distance <= MAX_DISTANCE:  # Adjust MAX_DISTANCE as needed
                    graph[v1][v2] = distance

    return frame, coordinates, graph

# Loop over frames from the video stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Detect the green rectangle (robot)
    robot = detect_robot(frame)

    # Detect the white balls and get valid contour coordinates and graph
    frame, white_ball_coords, graph = detect_white_balls(frame)

    # Find the starting point on the table tennis table
    start = robot[:2] if robot is not None else None

    # Check if the starting point was found
    if start is None:
        print("Starting point not found. Adjust camera position or choose a different starting point.")
        break

    # Find the shortest path to the closest ball using Dijkstra's algorithm
    distances = dijkstra(graph, start)
    closest_ball = min(distances, key=distances.get)

    # Draw lines from the robot to each white ball
    for ball_coord in white_ball_coords:
        cv2.line(frame, start, ball_coord, (255, 0, 0), 2)

    # Display the original frame with bounding boxes and lines
    cv2.imshow('Frame', frame)

    # Wait
    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key is pressed, break from the loop
    if key == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()
