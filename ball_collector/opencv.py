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
    #hsv_values = {'hmin': 129, 'smin': 0, 'vmin': 97, 'hmax': 180, 'smax': 197, 'vmax': 255}
    hsv_values = {'hmin': 150, 'smin': 50, 'vmin': 50, 'hmax': 180, 'smax': 255, 'vmax': 255}
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
"""        
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

    return frame, centers, graph"""

class Ball:
    x = None
    y = None
    radius = None

    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

class Point:
    x = None
    y = None

    def __init__(self, x, y):
        self.x = x
        self.y = y

cam_height = 163.5
robot_height = 15.5
ball_height = 4
goal_height = 7.5
obstacle_height = 3.5
cam_robot_scale = cam_height / robot_height
cam_ball_scale = cam_height / ball_height
cam_goal_scale = cam_height / goal_height
cam_obstacle_scale = cam_height / obstacle_height


def point_correction(cam_cen, bl_point):
    x_correction = round((cam_cen.x - bl_point.x) / cam_robot_scale)
    y_correction = round((cam_cen.y - bl_point.y) / cam_robot_scale)
    px = bl_point.x + x_correction
    py = bl_point.y + y_correction
    p = point.Point(px, py)
    return p


def goal_cen_correction(cam_cen, goal_cen):
    x_correction = round((cam_cen.x - goal_cen.x) / cam_goal_scale)
    y_correction = round((cam_cen.y - goal_cen.y) / cam_goal_scale)
    px = goal_cen.x + x_correction
    py = goal_cen.y + y_correction
    p = point.Point(px, py)
    return p


def obstacle_cen_correction(cam_cen, obstacle_cen):
    x_correction = round((cam_cen.x - obstacle_cen.x) / cam_obstacle_scale)
    y_correction = round((cam_cen.y - obstacle_cen.y) / cam_obstacle_scale)
    px = obstacle_cen.x + x_correction
    py = obstacle_cen.y + y_correction
    p = point.Point(px, py)
    return p


def ball_cen_correction(cam_cen, ball_cen):
    x_correction = round((cam_cen.x - ball_cen.x) / cam_ball_scale)
    y_correction = round((cam_cen.y - ball_cen.y) / cam_ball_scale)
    px = round(ball_cen.x + x_correction / 2)
    py = round(ball_cen.y + y_correction / 2)
    p = Point(px, py)
    return p

tempBall = []
def getBalls(img, polygon_pts):
    global tempBall
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.medianBlur(gray, 3)
    rows = gray.shape[1]
    # Original:
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10, param1=500, param2=26, minRadius=1, maxRadius=20)

    # Mere følsom:
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10, param1=500, param2=20, minRadius=1, maxRadius=20)
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10, param1=500, param2=15, minRadius=1, maxRadius=20)
    # print(circles)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        tempBall = []
        for i in circles[0, :]:
            center = (i[0], i[1])
            if cv2.pointPolygonTest(np.array(polygon_pts), center, False) >= 0:
                # circle center
                # cv2.circle(img, center, 1, (0, 100, 100), 5)
                # # circle outline
                # radius = i[2]
                # cv2.circle(img, center, radius, (255, 0, 255), 3)
                #singleBall = Ball(i[0], i[1], i[2])
                x_val = np.amax(img, axis=0)
                y_val = np.amax(img, axis=1)
                x_val = round(len(x_val) / 2)
                y_val = round(len(y_val) / 2)
                camera_center = Point(x_val, y_val)
                #corrected_ball_center = ball_cen_correction(camera_center, singleBall)
                #singleBall.x = int(corrected_ball_center.x)
                #singleBall.y = int(corrected_ball_center.y)
                # print("correctedball_X: " + str(corrected_ball_center.x))
                # print("correctedball_Y: " + str(corrected_ball_center.y))
                tempBall.append(center)
    else:
        tempBall.clear()
        return tempBall

    return tempBall


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

    return mask

def calculate_robotangle(robot_position, pink_position):
    # Calculate vectors
    robot_vector = np.array([robot_position[0] - pink_position[0], robot_position[1] - pink_position[1]])
    
    # Calculate magnitudes
    robot_magnitude = np.linalg.norm(robot_vector)
    pink_magnitude = np.linalg.norm(pink_position)
    
    # Avoid division by zero
    if robot_magnitude == 0 or pink_magnitude == 0:
        return 0
    
    # Calculate dot product
    dot_product = np.dot(robot_vector, pink_position)
    
    # Calculate cross product
    cross_product = np.cross(robot_vector, pink_position)
    
    # Calculate cosine of angle
    cos_angle = dot_product / (robot_magnitude * pink_magnitude)
    
    # Avoid values out of range due to floating-point inaccuracies
    cos_angle = np.clip(cos_angle, -1, 1)
    
    # Calculate the angle in radians
    angle_rad = np.arccos(cos_angle)
    
    # Calculate the angle in degrees
    angle_deg = np.degrees(angle_rad)
    
    # Determine the sign of the angle based on the cross product
    if cross_product > 0:
        angle_deg = -angle_deg
    
    return angle_deg

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

def calculate_angle4(robot_position, pink_position, closest_ball_position, fourth_position):
    import numpy as np

    # Calculate vectors
    robot_vector = np.array([robot_position[0] - pink_position[0], robot_position[1] - pink_position[1]])
    ball_vector = np.array([closest_ball_position[0] - pink_position[0], closest_ball_position[1] - pink_position[1]])
    fourth_vector = np.array([fourth_position[0] - pink_position[0], fourth_position[1] - pink_position[1]])
    
    # Calculate dot products
    dot_product1 = np.dot(robot_vector, ball_vector)
    dot_product2 = np.dot(ball_vector, fourth_vector)
    
    # Calculate magnitudes
    robot_magnitude = np.linalg.norm(robot_vector)
    ball_magnitude = np.linalg.norm(ball_vector)
    fourth_magnitude = np.linalg.norm(fourth_vector)
    
    # Avoid division by zero
    if robot_magnitude == 0 or ball_magnitude == 0 or fourth_magnitude == 0:
        return 0
    
    # Calculate cosines of angles
    cos_angle1 = dot_product1 / (robot_magnitude * ball_magnitude)
    cos_angle2 = dot_product2 / (ball_magnitude * fourth_magnitude)
    
    # Avoid values out of range due to floating-point inaccuracies
    cos_angle1 = np.clip(cos_angle1, -1, 1)
    cos_angle2 = np.clip(cos_angle2, -1, 1)
    
    # Calculate the angles in degrees
    angle1 = np.degrees(np.arccos(cos_angle1))
    angle2 = np.degrees(np.arccos(cos_angle2))
    
    # Determine the signs of the angles
    cross_product1 = np.cross(robot_vector, ball_vector)
    cross_product2 = np.cross(ball_vector, fourth_vector)
    
    if cross_product1 < 0:
        angle1 = -angle1
    
    if cross_product2 < 0:
        angle2 = -angle2
    
    # Calculate the average angle
    average_angle = (angle1 + angle2) / 2
    
    return average_angle

global goal
goal = False
getdistance = 100000
angle_checked = False
turnback = 100000
global goal2
@app.route('/', methods=['GET'])
def get_angle():
    global angle_checked  # Declare angle_checked and angle_checked2 as global
    test = 0
    switch_cases = {
        'turnback': lambda: jsonify({'turnback': turnback}),
        'goal_point': lambda: jsonify({'goal_point': robo_angle}),
        #'goal_right': lambda: jsonify({'goal_right': robo_angle}),
        #'goal_left': lambda: jsonify({'goal_left': robo_angle}),
        'onpoint': lambda: jsonify({'onpoint': (getdistance/2) + test}),
        'right': lambda: jsonify({'right': robo_angle}),
        'left': lambda: jsonify({'left': robo_angle}),
        'wait': lambda: jsonify({'wait': robo_angle})
    }

    case = 'wait'  # Default case
    goal2 = False
    if robo_angle <= -40 and robo_angle >= -35 and goal:
        goal2 = True
    print("robot angle =", robo_angle)
    if turnback <= 100:
        case = 'turnback'
    elif -1 <= robo_angle <= 1 and goal:
        print("GOOOOOOOOOAL")
        case = 'goal_point'
    #elif robo_angle > 1 and goal:
        #case = 'goal_left'
    #elif robo_angle < -1 and goal:
        #case = 'goal_right'
    elif -1 <= robo_angle <= 1:
        print("Aligned. Moving towards the ball...")
        print("distance =", getdistance)
        if getdistance <= 150:
            test = 15
        case = 'onpoint'
    elif robo_angle > 1:
        print("Turning right to align with the ball...", robo_angle)
        case = 'right'
    elif robo_angle < -1:
        print("Turning left to align with the ball...", robo_angle)
        case = 'left'

    return switch_cases[case]()  # Execute the corresponding switch case

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
Coordinates at click: 1617, 70
Coordinates at click: 1630, 948
Coordinates at click: 405, 929
Coordinates at click: 445, 64"""
hardcoded_coordinates = [(349, 85), (362, 1001), (1582, 957), (1539, 96)]

def draw_circles_at_coordinates(frame):
    # Define a list of hardcoded coordinates
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Convert hardcoded coordinates to a numpy array
    polygon_pts = np.array(hardcoded_coordinates, dtype=np.int32)
    polygon_pts = polygon_pts.reshape((-1, 1, 2))
    
    # Draw the border (polygon) by connecting coordinates
    cv2.polylines(frame, [polygon_pts], isClosed=True, color=(255, 0, 0), thickness=2)

    return polygon_pts  # Return the polygon points for later use


def draw_line(image, start, end, color, thickness=2):
    height, width = image.shape[:2]

    # Calculate the line endpoints that extend to the edges of the frame
    x1, y1 = start
    x2, y2 = end

    if x1 == x2:  # Vertical line
        x1 = x2 = max(0, min(x1, width - 1))
        y1 = 0
        y2 = height - 1
    else:
        m = (y2 - y1) / (x2 - x1)  # Slope
        b = y1 - m * x1  # Intercept

        x1 = 0
        y1 = int(b)
        x2 = width - 1
        y2 = int(m * x2 + b)

    cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    tempangle = calculate_robotangle(start, end)
    cv2.putText(frame, "Robotangle: {:.2f}".format(tempangle), (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)  # Draw lines from the robot to each white ball

# Loop over frames from the video stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    cap.set(cv2.CAP_PROP_FPS, 30)

    #frame = white_balance(frame)
    # Draw circles and get the polygon points
    polygon_pts = draw_circles_at_coordinates(frame)
    polygon_pts1 = draw_circles_at_redcross(frame)

    # Detect the green rectangle (robot)
    robot = detect_robot(frame)
    
    # Detect the white balls and get valid contour coordinates and graph
    #frame, ball_centers, graph = detect_white_balls(frame, polygon_pts)

    ball_centers = getBalls(frame, polygon_pts)

    pink = detect_pink(frame)

    #red = detect_red(frame)
    
    
    # Find the starting point on the table tennis table
    start = robot[:2] if robot is not None else None
    
    # Check if the starting point was found
    if start is None:
        print("Starting point not found. Adjust camera position or choose a different starting point.")
    else:

        draw_line(frame, start, pink, (0, 255, 0), 2)

        # Find the frame center
        frame_height, frame_width = frame.shape[:2]
        frame_center = (frame_width // 2, frame_height // 2)
        #right_edge = (frame_width - 100, frame_center[1])
        # left_edge = (700, frame_center[1])
        # # Calculate the intermediate point before the left edge
        # intermediate_point = (int(left_edge[0] * 0.5), left_edge[1])
        # # Draw a line from the frame center to the intermediate point (in yellow)
        # cv2.line(frame, frame_center, intermediate_point, (0, 255, 255), 2)
        # # Draw a line from the intermediate point to the left edge (in yellow)
        # cv2.line(frame, intermediate_point, left_edge, (0, 255, 255), 2)
        # cv2.circle(frame, intermediate_point, 5, (0, 255, 255), -1)
        # # Mark the left edge as the goal (draw a circle)
        # cv2.circle(frame, left_edge, 5, (0, 255, 255), -1)

        # Mark the half of hardcoded coordinates as the goal (draw a circle)
        # Calculate the center of the first two coordinates
        center_x = int((hardcoded_coordinates[0][0] + hardcoded_coordinates[1][0]) / 2)
        center_y = int((hardcoded_coordinates[0][1] + hardcoded_coordinates[1][1]) / 2)
        # Draw a circle at the center of the first two coordinates
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
    
        # Calculate the coordinates 200 pixels before the center
        prev_center_x = center_x + 330        
        # Draw a circle at the previous center coordinates
        cv2.circle(frame, (prev_center_x, center_y), 5, color=(0, 0, 255), thickness=-1)
        cv2.line(frame, (prev_center_x, center_y), (center_x, center_y), (0, 0, 255), 2)
    
        # Calculate the coordinates 200 pixels before the center
        prev_center_x2 = prev_center_x + 330        
        # Draw a circle at the previous center coordinates
        cv2.circle(frame, (prev_center_x2, center_y - 10), 5, color=(0, 0, 255), thickness=-1)
        cv2.line(frame, (prev_center_x, center_y), (center_x, center_y), (0, 0, 255), 2)
    
        # Filter ball_centers - keep only those that are inside the polygon
        #inside_ball_centers = [pt for pt in ball_centers if cv2.pointPolygonTest(polygon_pts, pt, False) >= 0]

        # Find the shortest path to the closest ball using Dijkstra's algorithm
        current_time = time.time()
        #if current_time - last_update_time >= 4:
        #closest_ball = closest_node(start, ball_centers)
            #last_update_time = current_time
            #closest_ball = closest_node(start, ball_centers)
        
        testdist = cv2.pointPolygonTest(polygon_pts, tuple(start), True)
        if testdist >= 0 and testdist <= 100:
            turnback = testdist
        else:
            turnback = 10000

        if len(ball_centers) <= 0:
            closest_ball = (prev_center_x, center_y)
            #print(distance(start, closest_ball))
            robo_angle = angle = calculate_angle(robot, pink, closest_ball)
            #print(robo_angle)
            getdistance = distance(start, closest_ball)
            if getdistance < 50:
                robo_angle = angle = calculate_angle(robot, pink, (prev_center_x2, center_y - -30))
                goal = True
                # if robo_angle > -35 and robo_angle < -40:
                #     print("GOAL2 TRUE!!!!!!!!!!!!!!!!!!!!!!!")
                #     goal2 = True
        else:
            closest_ball = closest_node(start, ball_centers)
            robo_angle = angle = calculate_angle(robot, pink, closest_ball)
            getdistance = distance(start, closest_ball)
            goal = False
            goal2 = False

        if closest_ball is not None and start is not None and frame is not None and pink is not None:
            if angle is not None:
                cv2.putText(frame, "Angle: {:.2f}".format(angle), (frame_center), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)  # Draw lines from the robot to each white ball
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