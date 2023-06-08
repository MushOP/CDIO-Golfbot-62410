import cv2
import numpy as np
import heapq
import math
from flask import Flask, jsonify
app = Flask(__name__)
import threading

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

# Function to detect the green rectangle (robot)
def detect_pink(frame):
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the HSV values for the green rectangle
    hsv_values = {'hmin': 129, 'smin': 0, 'vmin': 97, 'hmax': 180, 'smax': 197, 'vmax': 255}
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
        return center

def detect_white_balls(frame):
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range of white color in HSV
    hsv_values = {'hmin': 0, 'smin': 0, 'vmin': 218, 'hmax': 180, 'smax': 20, 'vmax': 255}
    lower_white = np.array([hsv_values['hmin'], hsv_values['smin'], hsv_values['vmin']])
    upper_white = np.array([hsv_values['hmax'], hsv_values['smax'], hsv_values['vmax']])


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
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Extract valid contour coordinates
    #coordinates = [(cv2.boundingRect(contour)[:2]) for contour in valid_contours]

    # Extract valid contour coordinates and centers
    coordinates = []
    centers = []
    for contour in valid_contours:
        x, y, w, h = cv2.boundingRect(contour)
        coordinates.append((x, y, w, h))
        center = (x + w // 2, y + h // 2)  # Calculate center of the bounding box
        centers.append(center)

    # Create a graph with valid contour coordinates as vertices
    graph = {coord: {} for coord in coordinates}

    # Connect each vertex in the graph with its neighboring vertices
    for v1 in graph:
        for v2 in graph:
            if v1 != v2:
                distance = np.sqrt((v2[0] - v1[0]) ** 2 + (v2[1] - v1[1]) ** 2)
                if distance <= MAX_DISTANCE:  # Adjust MAX_DISTANCE as needed
                    graph[v1][v2] = distance

    return frame, coordinates, centers, graph

def detect_red(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper red color ranges in HSV
    hsv_values = {'hmin': 0, 'smin': 150, 'vmin': 175, 'hmax': 10, 'smax': 255, 'vmax': 255}
    """
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    """
    lower_red = np.array([hsv_values['hmin'], hsv_values['smin'], hsv_values['vmin']])
    upper_red = np.array([hsv_values['hmax'], hsv_values['smax'], hsv_values['vmax']])
    # Threshold the HSV image to get only the red color
    #mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    #mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine the masks
    #mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    # Apply morphological operations to remove noise from the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Changed to MORPH_CLOSE

    # Find contours of the red regions
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)  # Changed to RETR_LIST

    # Draw contours on the frame
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    return frame

def calculate_angle(green_robot, pink_robot, ball_center):
    if green_robot is not None and pink_robot is not None and ball_center is not None:
        green_center = green_robot
        pink_center = pink_robot

        # Calculate the angle between the green robot and the ball center
        angle1 = math.atan2(ball_center[1] - green_center[1], ball_center[0] - green_center[0])

        # Calculate the angle between the green robot and the pink robot
        angle2 = math.atan2(green_center[1] - pink_center[1], green_center[0] - pink_center[0])

        # Calculate the angle between the green robot's front and the line connecting it to the ball center
        angle_degrees = math.degrees(angle1 - angle2)

        # Adjust the angle to be between -180 and 180 degrees
        if angle_degrees > 180:
            angle_degrees -= 360
        elif angle_degrees < -180:
            angle_degrees += 360

        return -angle_degrees  # Reverse the angle when the ball is in front of the green robot
    else:
        return None
global goal
goal = False
@app.route('/', methods=['GET'])
def get_angle():
    #print(goal)
    if robo_angle >= -145 and robo_angle <= 155 and goal:
        print("hit goal")
        return jsonify({'goal_point': robo_angle - 180}) 
    elif robo_angle >= -7 and robo_angle <= 7:
        #print("hit on")
        return jsonify({'onpoint': robo_angle})
    elif robo_angle < 0:
        return jsonify({'left': robo_angle})
    elif robo_angle > 0:
        return jsonify({'right': robo_angle})
    else:
        return jsonify({'idk': robo_angle})

# Function to run the Flask app in a separate thread
def run_flask_app():
    app.run(host='0.0.0.0', port=5001)

# Start the Flask app on a separate thread
flask_thread = threading.Thread(target=run_flask_app)
flask_thread.start()

global robo_angle

# Loop over frames from the video stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Detect the green rectangle (robot)
    robot = detect_robot(frame)

    # Detect the white balls and get valid contour coordinates and graph
    frame, white_ball_coords, ball_centers, graph = detect_white_balls(frame)

    pink = detect_pink(frame)

    red = detect_red(frame)
    
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
        left_edge = (500, frame_center[1])
        # Calculate the intermediate point before the left edge
        intermediate_point = (int(left_edge[0] * 0.5), left_edge[1])
        # Draw a line from the frame center to the intermediate point (in yellow)
        cv2.line(frame, frame_center, intermediate_point, (0, 255, 255), 2)
        # Draw a line from the intermediate point to the left edge (in yellow)
        #cv2.line(frame, intermediate_point, left_edge, (0, 255, 255), 2)
        cv2.circle(frame, intermediate_point, 5, (0, 255, 255), -1)
        # Mark the left edge as the goal (draw a circle)
        cv2.circle(frame, left_edge, 5, (0, 255, 255), -1)
        
        

        # Find the shortest path to the closest ball using Dijkstra's algorithm
        closest_ball = closest_node(start, ball_centers)
        robo_angle = 0
        if len(ball_centers) <= 0:
            closest_ball = left_edge
            #print(distance(start, closest_ball))
            robo_angle = angle = calculate_angle(robot, pink, closest_ball)
            print(robo_angle)
            if distance(start, closest_ball) < 160 and robo_angle >= -145 and robo_angle <= -155:
                goal = True
        #closest_ball = left_edge
        # Draw a line from the starting point to the closest ball
        #print(closest_ball)
        if closest_ball is not None and start is not None and frame is not None:
            cv2.line(frame, start, closest_ball, (255, 255, 0), 2)

            robo_angle = angle = calculate_angle(robot, pink, closest_ball)
            
            if angle is not None:
                cv2.putText(frame, "Angle: {:.2f}".format(angle), (start), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)  # Draw lines from the robot to each white ball
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