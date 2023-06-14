import cv2
import numpy as np
import heapq
import math
from flask import Flask, jsonify
app = Flask(__name__)
import threading
import time
last_update_time = time.time() - 3

# Create a VideoCapture object to read from the camera
cap = cv2.VideoCapture(0)
MAX_DISTANCE = 1000000

def closest_node(node, nodes):
    pt = []
    dist = 9999999

    for n in nodes:
        if distance(node, n) <= dist:
            dist = distance(node, n)
            pt = n

    return pt

def distance(pt1, pt2):
    pt1 = np.array((pt1[0], pt1[1]))
    pt2 = np.array((pt2[0], pt2[1]))
    return np.linalg.norm(pt1 - pt2)

# Function to detect the green rectangle (robot)
def detect_robot(frame):
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the HSV values for the green rectangle
    hsv_values = {'hmin': 29, 'smin': 45, 'vmin': 27, 'hmax': 92, 'smax': 255, 'vmax': 255}
    #hsv_values = {'hmin': 40, 'smin': 75, 'vmin': 20, 'hmax': 80, 'smax': 255, 'vmax': 255}
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

# Function to detect the green rectangle (robot)
def detect_pink(frame):
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the HSV values for the green rectangle
    hsv_values = {'hmin': 129, 'smin': 0, 'vmin': 97, 'hmax': 180, 'smax': 197, 'vmax': 255}
    #hsv_values = {'hmin': 150, 'smin': 50, 'vmin': 50, 'hmax': 180, 'smax': 255, 'vmax': 255}
    #hsv_values = {'hmin': 150, 'smin': 100, 'vmin': 100, 'hmax': 200, 'smax': 255, 'vmax': 255}
    #hsv_values = {'hmin': 170, 'smin': 50, 'vmin': 50, 'hmax': 180, 'smax': 255, 'vmax': 255}
    lower_green = np.array([hsv_values['hmin'], hsv_values['smin'], hsv_values['vmin']])
    upper_green = np.array([hsv_values['hmax'], hsv_values['smax'], hsv_values['vmax']])

    # Threshold the HSV image to get only green colors
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Apply morphological operations to remove noise
    kernel = np.ones((10, 10), np.uint8)
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
        cv2.drawContours(frame, [box], 0, (255, 192, 203), 3)
        center = np.mean(box, axis=0)
        return center
        
def detect_white_balls(frame, polygon_pts):
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of white color in HSV
    hsv_values = {'hmin': 0, 'smin': 0, 'vmin': 220, 'hmax': 220, 'smax': 30, 'vmax': 255}
    lower_white = np.array([hsv_values['hmin'], hsv_values['smin'], hsv_values['vmin']])
    upper_white = np.array([hsv_values['hmax'], hsv_values['smax'], hsv_values['vmax']])

    # Threshold the HSV image to get only white colors
    mask = cv2.inRange(hsv, lower_white, upper_white)

    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours of white regions
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area, shape, and circularity
    min_area = 100  # Minimum area threshold for contours
    max_area = 2000  # Maximum area threshold for contours
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        # Check if the ball center is inside the polygon
        if (min_area < area < max_area and 0.5 < circularity < 1.2):  # adjust circularity range as needed
            valid_contours.append(contour)

    # Draw bounding circles around valid contours
    # Extract valid contour coordinates and centers
    coordinates = []
    centers = []
    for contour in valid_contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        #cv2.circle(frame, center, radius, (0, 255, 255), 2)
        # Check if the ball center is inside the polygon
        if cv2.pointPolygonTest(np.array(polygon_pts), center, False) >= 0:
            cv2.circle(frame, center, radius, (0, 255, 255), 2)
            coordinates.append((int(x - radius), int(y - radius), int(radius * 2), int(radius * 2)))
            centers.append(center)

    # Create a graph with valid contour coordinates as vertices
    graph = {coord: {} for coord in coordinates}

    # Connect each vertex in the graph with its neighboring vertices
    MAX_DISTANCE = 50  # adjust this as needed
    for v1 in graph:
        for v2 in graph:
            if v1 != v2:
                distance = np.sqrt((v2[0] - v1[0]) ** 2 + (v2[1] - v1[1]) ** 2)
                if distance <= MAX_DISTANCE:
                    graph[v1][v2] = distance

    return frame, centers, graph

def detect_red(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper red color ranges in HSV
    hsv_values = {'hmin': 0, 'smin': 170, 'vmin': 175, 'hmax': 10, 'smax': 255, 'vmax': 255}
    lower_red = np.array([hsv_values['hmin'], hsv_values['smin'], hsv_values['vmin']])
    upper_red = np.array([hsv_values['hmax'], hsv_values['smax'], hsv_values['vmax']])
    
    # Threshold the HSV image to get only the red color
    mask = cv2.inRange(hsv, lower_red, upper_red)
    
    # Apply morphological operations to remove noise from the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours of the red regions
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop through the contours
    for contour in contours:
        # Check contour size, if it's too small, skip it
        if cv2.contourArea(contour) < 100:
            continue

        # Find the minimum enclosing circle for each contour
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        
        # Calculate the coordinates of each corner of a square circumscribing the circle
        half_side_length = radius * np.sqrt(2)
        top_left = (int(x - half_side_length), int(y - half_side_length))
        top_right = (int(x + half_side_length), int(y - half_side_length))
        bottom_left = (int(x - half_side_length), int(y + half_side_length))
        bottom_right = (int(x + half_side_length), int(y + half_side_length))
        
        # Draw a small circle at each corner of the square
        cv2.circle(frame, top_left, 5, (0, 255, 255), -1)
        cv2.circle(frame, top_right, 5, (0, 255, 255), -1)
        cv2.circle(frame, bottom_left, 5, (0, 255, 255), -1)
        cv2.circle(frame, bottom_right, 5, (0, 255, 255), -1)

    return frame

def calculate_angle(robot_position, pink_position, closest_ball_position):
    # Calculate vectors
    robot_vector = np.array([robot_position[0] - pink_position[0], robot_position[1] - pink_position[1]])
    ball_vector = np.array([closest_ball_position[0] - pink_position[0], closest_ball_position[1] - pink_position[1]])
    
    # Calculate dot product
    dot_product = np.dot(robot_vector, ball_vector)
    
    # Calculate cross product
    cross_product = np.cross(robot_vector, ball_vector)
    
    # Calculate magnitudes
    robot_magnitude = np.linalg.norm(robot_vector)
    ball_magnitude = np.linalg.norm(ball_vector)
    
    # Avoid division by zero
    if robot_magnitude == 0 or ball_magnitude == 0:
        return 0
    
    # Calculate cosine of angle
    cos_angle = dot_product / (robot_magnitude * ball_magnitude)
    
    # Avoid values out of range due to floating-point inaccuracies
    cos_angle = np.clip(cos_angle, -1, 1)
    
    # Calculate the angle in degrees
    angle = np.degrees(np.arccos(cos_angle))
    
    # Determine the sign of the angle
    if cross_product < 0:
        angle = -angle

    return angle

global goal
goal = False
getdistance = 100000
angle_checked = False
@app.route('/', methods=['GET'])
def get_angle():
    global angle_checked  # Declare angle_checked and angle_checked2 as global
    test = 0
    #if not angle_checked:  # First angle check
    if -1 <= robo_angle <= 1:
        #angle_checked = True
        #if getdistance > 300:
        print("Aligned. Moving towards the ball...")
        print("distance = ", getdistance)
        if getdistance <= 150:
            test = 25
        return jsonify({'onpoint': (getdistance/2) + 10})
    elif robo_angle > 1:
        print("Turning right to align with the ball...")
        return jsonify({'right': robo_angle})
    elif robo_angle < -1:
        print("Turning left to align with the ball...")
        return jsonify({'left': robo_angle})

    return jsonify({'wait': robo_angle})

# Function to run the Flask app in a separate thread
def run_flask_app():
    app.run(host='0.0.0.0', port=5001)

# Start the Flask app on a separate thread
flask_thread = threading.Thread(target=run_flask_app)
flask_thread.start()

global robo_angle

# Function to apply simple white balancing to the frame
def white_balance(frame):
    result = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return result


manMode = False
selectCount = 0
# Field dimensions
width = 750
height = 500
corners = [[0, 0], [750, 0], [0, 500], [750, 500]]
goals = [[0, 250], [750, 250]]

def transform(frame, frameCount):
    if (not manMode and frameCount % 20 == 0):
        corners = getCorners(frame)
        goals = getGoals(corners)

    if (corners != None):
        # Coordinates for the corners
        oldCoordinates = np.float32([corners[0], corners[1],
                                        corners[2], corners[3]])

        # New coordinates for the corners
        newCoordinates = np.float32([[0, 0], [width, 0],
                                        [0, height], [width, height]])

        # Transform image
        matrix = cv2.getPerspectiveTransform(
            oldCoordinates, newCoordinates)
        transformed = cv2.warpPerspective(
            frame, matrix, (width, height))
        # Create circles to indicate detected corners and goals
        frame = cv2.circle(frame, tuple(
            corners[0]), 20, (255, 0, 0), 2)
        frame = cv2.circle(frame, tuple(
            corners[1]), 20, (255, 0, 0), 2)
        frame = cv2.circle(frame, tuple(
            corners[2]), 20, (255, 0, 0), 2)
        frame = cv2.circle(frame, tuple(
            corners[3]), 20, (255, 0, 0), 2)
        frame = cv2.circle(frame, tuple(goals[0]), 20, (255, 0, 0), 2)
        frame = cv2.circle(frame, tuple(goals[1]), 20, (255, 0, 0), 2)
        return transformed
import numpy as np
import cv2

def getcorners(frame):
    # Define lower/upper color field for red
    lower_red = np.array([0, 0, 200], dtype="uint8")
    upper_red = np.array([100, 100, 255], dtype="uint8")
    
    # Frame dimensions
    height, width = frame.shape[:2]

    # Create a mask for red color
    red_mask = cv2.inRange(frame, lower_red, upper_red)

    # Filter out everything that is not red
    wall = cv2.bitwise_and(frame, frame, mask=red_mask)
    
    # Convert to grayscale
    wall = cv2.cvtColor(wall, cv2.COLOR_BGR2GRAY)

    # Find corners
    dest = cv2.cornerHarris(np.float32(wall), 20, 9, 0.14)
    dest = cv2.dilate(dest, None)

    # Threshold for detecting corners
    thresh = 0.01 * dest.max()
    
    # Get the coordinates of the corners
    corners_y, corners_x = np.where(dest > thresh)

    # Calculate center
    center_x, center_y = width / 2, height / 2
    
    # Initialize corner coordinates
    upper_left = [width, height]
    upper_right = [0, height]
    lower_left = [width, 0]
    lower_right = [0, 0]

    # Iterate through the detected corners and update the corner coordinates
    for x, y in zip(corners_x, corners_y):
        if x < center_x and y < center_y:
            upper_left = [min(upper_left[0], x), min(upper_left[1], y)]
        elif x > center_x and y < center_y:
            upper_right = [max(upper_right[0], x), min(upper_right[1], y)]
        elif x < center_x and y > center_y:
            lower_left = [min(lower_left[0], x), max(lower_left[1], y)]
        else:
            lower_right = [max(lower_right[0], x), max(lower_right[1], y)]
            
    # Draw circles at the corners
    cv2.circle(frame, tuple(map(int, upper_left)), 20, (255, 0, 0), 2)
    cv2.circle(frame, tuple(map(int, upper_right)), 20, (255, 0, 0), 2)
    cv2.circle(frame, tuple(map(int, lower_left)), 20, (255, 0, 0), 2)
    cv2.circle(frame, tuple(map(int, lower_right)), 20, (255, 0, 0), 2)

    # Return the coordinates of the corners
    return [upper_left, upper_right, lower_left, lower_right]


def getGoals(corners):
    goal_left = [0, 99999]
    goal_right = [99999, 0]
    upper_left = corners[0]
    upper_right = corners[1]
    lower_left = corners[2]
    lower_right = corners[3]

    goal_left[0] = int((upper_left[0] + lower_left[0]) / 2)
    goal_left[1] = int((upper_left[1] + lower_left[1]) / 2)
    goal_right[0] = int((upper_right[0] + lower_right[0]) / 2)
    goal_right[1] = int((upper_right[1] + lower_right[1]) / 2)

    return [goal_left, goal_right]
    """
Coordinates at click: 1663, 183
Coordinates at click: 1688, 1039
Coordinates at click: 450, 1019
Coordinates at click: 526, 134"""

def draw_circles_at_coordinates(frame):
    # Define a list of hardcoded coordinates
    hardcoded_coordinates = [(1663, 183), (1688, 1039), (450, 1019), (526, 134)]
    
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Convert hardcoded coordinates to a numpy array
    polygon_pts = np.array(hardcoded_coordinates, dtype=np.int32)
    polygon_pts = polygon_pts.reshape((-1, 1, 2))
    
    # Draw the border (polygon) by connecting coordinates
    cv2.polylines(frame, [polygon_pts], isClosed=True, color=(255, 0, 0), thickness=2)

    return polygon_pts  # Return the polygon points for later use

# Loop over frames from the video stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    #frame = white_balance(frame)
    # Draw circles and get the polygon points
    polygon_pts = draw_circles_at_coordinates(frame)

    # Detect the green rectangle (robot)
    robot = detect_robot(frame)

    # Detect the white balls and get valid contour coordinates and graph
    frame, ball_centers, graph = detect_white_balls(frame, polygon_pts)

    pink = detect_pink(frame)

    #red = detect_red(frame)
    
    # Find the starting point on the table tennis table
    start = robot[:2] if robot is not None else None

    # Check if the starting point was found
    if start is None:
        print("Starting point not found. Adjust camera position or choose a different starting point.")
    else:  
        # Find the frame center
        frame_height, frame_width = frame.shape[:2]
        frame_center = (frame_width // 2, frame_height // 2)
        #right_edge = (frame_width - 100, frame_center[1])
        left_edge = (700, frame_center[1])
        # Calculate the intermediate point before the left edge
        intermediate_point = (int(left_edge[0] * 0.5), left_edge[1])
        # Draw a line from the frame center to the intermediate point (in yellow)
        cv2.line(frame, frame_center, intermediate_point, (0, 255, 255), 2)
        # Draw a line from the intermediate point to the left edge (in yellow)
        #cv2.line(frame, intermediate_point, left_edge, (0, 255, 255), 2)
        cv2.circle(frame, intermediate_point, 5, (0, 255, 255), -1)
        # Mark the left edge as the goal (draw a circle)
        cv2.circle(frame, left_edge, 5, (0, 255, 255), -1)

        # Filter ball_centers - keep only those that are inside the polygon
        inside_ball_centers = [pt for pt in ball_centers if cv2.pointPolygonTest(polygon_pts, pt, False) >= 0]
        
        # Find the shortest path to the closest ball inside the polygon using Dijkstra's algorithm
        closest_ball = closest_node(start, inside_ball_centers) if inside_ball_centers else None

        # Find the shortest path to the closest ball using Dijkstra's algorithm
        current_time = time.time()
        #if current_time - last_update_time >= 4:
        #closest_ball = closest_node(start, ball_centers)
            #last_update_time = current_time
            #closest_ball = closest_node(start, ball_centers)
        robo_angle = 0
        if len(ball_centers) <= 0:
            closest_ball = left_edge
            #print(distance(start, closest_ball))
            robo_angle = angle = calculate_angle(robot, pink, closest_ball)
            #print(robo_angle)
            getdistance = distance(start, closest_ball)
            if getdistance < 160:
                print("goooooal")
                goal = True
        #closest_ball = left_edge
        # Draw a line from the starting point to the closest ball
        #print(closest_ball)
        if closest_ball is not None and start is not None and frame is not None and pink is not None:
            cv2.line(frame, start, closest_ball, (255, 255, 0), 2)

            robo_angle = angle = calculate_angle(robot, pink, closest_ball)
            getdistance = distance(start, closest_ball)
            #print("distance = ", getdistance)

            if angle is not None:
                cv2.putText(frame, "Angle: {:.2f}".format(angle), (frame_center), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)  # Draw lines from the robot to each white ball
            #cv2.putText(frame, "Distance: {:.2f}".format(distance(start, closest_ball)), (start), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)    # Draw lines from the robot to each white ball
        #for ball_coord in white_ball_coords:
            #cv2.line(frame, start, ball_coord, (255, 0, 0), 2)
        #print("****", closest_ball)
        #print("****", white_ball_coords)
        cv2.line(frame, start, closest_ball, (255, 255, 0), 2)
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