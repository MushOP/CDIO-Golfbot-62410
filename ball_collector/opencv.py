import cv2
import numpy as np
import math
from flask import Flask, jsonify
app = Flask(__name__)
import threading
import time
import traceback
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

last_update_time = time.time()

# Create a VideoCapture object to read from the camera
cap = cv2.VideoCapture(0)
MAX_DISTANCE = 1000000

def closest_node(node, nodes):
    pt = []
    dist = 9999999

    for n in nodes:
        if distance(node, n['center']) <= dist:
            dist = distance(node, n['center'])
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

detected_balls = []
ball_id = 0  # Declare ball_id as a global variable

def is_inside_small_boxes(small_boxes, point):
    for box in small_boxes:
        top_left, bottom_right = box
        if (
            top_left[0] <= point[0] <= bottom_right[0]
            and top_left[1] <= point[1] <= bottom_right[1]
        ):
            print('Ball inside small box')
            return True
            
    return False

def getBalls(frame, polygon_pts, small_boxes, red_cross_polygon):
    global ball_id  # Access the global ball_id variable

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 10,
                               param1=500, param2=26, minRadius=1, maxRadius=20)
    
    detected_balls = []
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            if cv2.pointPolygonTest(np.array(polygon_pts), center, False) >= 0 and not any(cv2.pointPolygonTest(np.array(box_pts), center, False) > 0 for box_pts in small_boxes) and not cv2.pointPolygonTest(np.array(red_cross_polygon), center, False) > 0:
                ball_id = 1
                detected_balls.append({
                    'id': ball_id,
                    'center': center
                })
                cv2.circle(frame, center, 5, (0, 100, 100), 3)
    return detected_balls

def is_inside_rectangle(point, rect_center, width, height):
    x, y = point
    rect_x, rect_y = rect_center
    half_width = width // 2
    half_height = height // 2
    if (
        x >= rect_x - half_width and
        x <= rect_x + half_width and
        y >= rect_y - half_height and
        y <= rect_y + half_height
    ):
        return True
    return False

def detect_red(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper red color ranges in HSV
    hsv_values = {'hmin': 0, 'smin': 70, 'vmin': 50, 'hmax': 10, 'smax': 255, 'vmax': 255}
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

    return contours
def detect_red_cross(frame, polygon_pts):
    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the HSV values for the red color
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Threshold the HSV image to get only red colors
    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

    # Find contours of red regions
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the red cross
    red_cross_polygon = None
    for contour in contours_red:
        x, y, w, h = cv2.boundingRect(contour)
        x = x - 15
        y = y - 15
        w = w + 30
        h = h + 30
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)
        if cv2.pointPolygonTest(polygon_pts, (int(x), int(y)), False) < 0:
            continue
        # Adjust the threshold values as needed
        if aspect_ratio >= 0.4 and aspect_ratio <= 1.6 and area > 500:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            # Return the coordinates of the vertices of the bounding rectangle
            red_cross_polygon = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            break

    return red_cross_polygon

def red_contors(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_values = {'hmin': 0, 'smin': 150, 'vmin': 175, 'hmax': 10, 'smax': 255, 'vmax': 255}

    # Get a binary image isolating the red pixels
    lower_red = np.array([hsv_values['hmin'], hsv_values['smin'], hsv_values['vmin']])
    upper_red = np.array([hsv_values['hmax'], hsv_values['smax'], hsv_values['vmax']])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return red_contours

def calculate_robotangle(robot_position, pink_position):
    if any(val is None for val in (robot_position, pink_position)):
        return 0  # Skip drawing the line if any value is None

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
goforward = 100000
turnamt = 0
turnleft = 0
turnright = 0
turnbackforced = 10000
turngoalpoint = 0
pushstart = True 
global goal2
@app.route('/', methods=['GET'])
def get_angle():
    global angle_checked  # Declare angle_checked and angle_checked2 as global
    global turnamt, turnleft, turnright, pushstart, turngoalpoint
    test = 0
    switch_cases = {
        'backward': lambda: jsonify({'backward': turnback}),
        'goal_point': lambda: jsonify({'goal_point': robo_angle}),
        #'goal_right': lambda: jsonify({'goal_right': robo_angle}),
        #'goal_left': lambda: jsonify({'goal_left': robo_angle}),
        'onpoint': lambda: jsonify({'onpoint': (getdistance/2) + test}),
        'right': lambda: jsonify({'right': robo_angle}),
        'left': lambda: jsonify({'left': robo_angle}),
        'forward': lambda: jsonify({'forward': goforward}),
        'forward_cross': lambda: jsonify({'forward_cross': (getdistance/2)}),
        'wait': lambda: jsonify({'wait': robo_angle})
    }

    case = 'wait'  # Default case
    goal2 = False
    #print("robot angle =", robo_angle)
    if turnback <= 100 or turnbackforced <= 110:
        print("turnback!!!")
        if turnamt < 10:
            case = 'backward'
            turnamt += 1
        else:
            move_backward = False
    elif goforward <= 100:
        case = 'forward'
    elif pushstart:
        case = 'forward_cross'
    elif -0.5 <= robo_angle <= 0.5 and goal:
        print("GOOOOOOOOOAL")
        case = 'goal_point'
        turngoalpoint += 1
        turnamt = 0
    #elif robo_angle > 1 and goal:
        #case = 'goal_left'
    #elif robo_angle < -1 and goal:
        #case = 'goal_right'
    elif -1 <= robo_angle <= 1:
        print("Aligned. Moving towards the ball...")
        #print("distance =", getdistance, " test = ", test)
        if getdistance <= 200:
            print("HIT TEST")
            test = 75
        case = 'onpoint'
        turnamt = 0
    elif robo_angle > 1:
        print("Turning right to align with the ball...", robo_angle)
        case = 'right'
        turnamt = 0
        turnright += 1
        turnleft = 0
    elif robo_angle < -1:
        print("Turning left to align with the ball...", robo_angle)
        case = 'left'
        turnamt = 0
        turnright = 0
        turnleft += 1

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
#hardcoded_coordinates = [(466, 168), (470, 917), (1539, 905), (1486, 165)]
hardcoded_coordinates = [(394, 93), (390, 954), (1582, 1019), (1631, 105)]
def draw_circles_at_coordinates(frame):
    # Define a list of hardcoded coordinates
 
    # Get frame dimensions
    height, width = frame.shape[:2]

    # Define the size of the small box
    box_size = 80

    # Convert hardcoded coordinates to a numpy array
    polygon_pts = np.array(hardcoded_coordinates, dtype=np.int32)
    polygon_pts = polygon_pts.reshape((-1, 1, 2))
    # Draw the border (polygon) by connecting coordinates
    cv2.polylines(frame, [polygon_pts], isClosed=True, color=(255, 0, 0), thickness=2)

    # Draw small boxes at the corners
    small_boxes = []
    for coord in hardcoded_coordinates:
        top_left = (coord[0] - box_size, coord[1] - box_size)
        top_right = (coord[0] + box_size, coord[1] - box_size)
        bottom_right = (coord[0] + box_size, coord[1] + box_size)
        bottom_left = (coord[0] - box_size, coord[1] + box_size)
        box_pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
        box_pts = box_pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [box_pts], isClosed=True, color=(0, 0, 255), thickness=2)
        small_boxes.append(box_pts)

    return polygon_pts, small_boxes  # Return the polygon points for later use

def draw_line_points(frame):
    num = 180
    # Draw a line between the 1st and 2nd coordinates
    left_start = (hardcoded_coordinates[0][0] + num, hardcoded_coordinates[0][1] + num)
    left_end = (hardcoded_coordinates[1][0] + num, hardcoded_coordinates[1][1] - num)
    cv2.line(frame, left_start, left_end, color=(0, 255, 0), thickness=2)
    cv2.putText(frame, "left", left_start, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw a line between the 3rd and 4th coordinates
    right_start = (hardcoded_coordinates[2][0] - num, hardcoded_coordinates[2][1] - num)
    right_end = (hardcoded_coordinates[3][0] - num, hardcoded_coordinates[3][1] + num)
    cv2.line(frame, right_start, right_end, color=(0, 255, 0), thickness=2)
    cv2.putText(frame, "right", right_start, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw a line between the first and fourth coordinates
    upper_start = (hardcoded_coordinates[0][0] + num, hardcoded_coordinates[0][1] + num)
    upper_end = (hardcoded_coordinates[3][0] - num, hardcoded_coordinates[3][1] + num)
    cv2.line(frame, upper_start, upper_end, color=(0, 255, 0), thickness=2)
    cv2.putText(frame, "right", right_start, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw a line between the 2nd and 3rd coordinates
    down_start = (hardcoded_coordinates[1][0] + num, hardcoded_coordinates[1][1] - num)
    down_end = (hardcoded_coordinates[2][0] - num, hardcoded_coordinates[2][1] - num)
    cv2.line(frame, down_start, down_end, color=(0, 255, 0), thickness=2)
    cv2.putText(frame, "down", down_start, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return (left_start, left_end), (right_start, right_end), (upper_start, upper_end), (down_start, down_end)

def is_within_boundary(point, boundary_points):
    return cv2.pointPolygonTest(np.array(boundary_points), point, False) >= 0

def draw_line_to_nearest_boundary(frame, ball, boundaries):
    # Assume that ball is a tuple (x, y) representing the position of the ball
    # boundaries is a list of tuples (start_point, end_point) representing boundary lines
    
    min_distance = float("inf")
    nearest_boundary_start = None
    nearest_boundary_end = None
    
    for start, end in boundaries:
        nearest_point_on_line = get_nearest_point_on_line(ball, start, end)
        distance = np.sqrt((nearest_point_on_line[0] - ball[0]) ** 2 + (nearest_point_on_line[1] - ball[1]) ** 2)

        if distance < min_distance:
            min_distance = distance
            nearest_boundary_start = start
            nearest_boundary_end = end
    
    # Draw the line from ball to nearest boundary
    if nearest_boundary_start and nearest_boundary_end:
        line_length = 200  # Define the desired length of the line

        nearest_point_on_line = get_nearest_point_on_line(ball, nearest_boundary_start, nearest_boundary_end, line_length)
        #nearest_point_on_line[0][0] = int(nearest_point_on_line[0][0]) + 5
        
        # Draw the blue line from ball to nearest point on the boundary
        cv2.line(frame, ball, nearest_point_on_line, color=(255, 0, 0), thickness=2)
        
        # Draw the green boundary line
        cv2.line(frame, nearest_boundary_start, nearest_boundary_end, color=(0, 255, 0), thickness=2)
        
        return nearest_point_on_line
    
    return False

def get_nearest_point_on_line(point, line_start, line_end, line_length=200):
    line_vec = np.array(line_end) - np.array(line_start)
    point_vec = np.array(point) - np.array(line_start)
    line_len = np.dot(line_vec, line_vec)
    factor = np.dot(point_vec, line_vec) / line_len
    nearest_point = np.array(line_start) + factor * line_vec
    # Clamp factor between 0 and 1
    factor = max(0, min(1, factor))

    nearest_point = np.array(line_start) + factor * line_vec
    #nearest_point[0] += 80  # Add +10 to the x-coordinate of nearest_point
    return tuple(map(int, nearest_point))


def draw_line(image, start, end, color, thickness=2):
    if any(val is None for val in (start, end, image, color)):
        return  # Skip drawing the line if any value is None
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

closest_ball = None
def selected_ball(robot, balls):
    closest_ball = closest_node(robot, balls)
    return closest_ball

def calculate_distance_to_polygon(ball, polygon_pts):
    return cv2.pointPolygonTest(np.array(polygon_pts), ball, True)

previous_ball = None  # Initialize previous_ball variable before the loop
retry = True
distancecheck = False
distance_threshold = 50

start_time = time.time()
safepoint = False
prepoint = False
prepointmarked = False
forcedgotogoal = False
firstpushstart = False
red_cross_centers = None
while retry:
    try:
        # Loop over frames from the video stream
        while True:
            # Read a frame from the video stream
            ret, frame = cap.read()
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            #frame = white_balance(frame)
            # Draw circles and get the polygon points
            polygon_pts, small_boxes = draw_circles_at_coordinates(frame)
            #polygon_pts1 = draw_circles_at_redcross(frame)

            # Detect the green rectangle (robot)
            robot = detect_robot(frame)
            
            # Detect the white balls and get valid contour coordinates and graph
            #frame, ball_centers, graph = detect_white_balls(frame, polygon_pts)
            redcross = detect_red_cross(frame, polygon_pts)

            ball_centers = getBalls(frame, polygon_pts, small_boxes, redcross)
            #ball_centers = detect_white_balls(frame, polygon_pts)
            pink = detect_pink(frame)

            # Find the starting point on the table tennis table
            start = robot[:2] if robot is not None else None
            boundaries = draw_line_points(frame)
            #cross = detect_cross(frame)
            # Check if the starting point was found
            if start is None:
                print("Starting point not found. Adjust camera position or choose a different starting point.")
            else:
                draw_line(frame, start, pink, (0, 255, 0), 2)
                
                # Find the frame center
                frame_height, frame_width = frame.shape[:2]
                frame_center = (frame_width // 2, frame_height // 2)

                # Mark the half of hardcoded coordinates as the goal (draw a circle)
                # Calculate the center of the first two coordinates
                center_x = int((hardcoded_coordinates[0][0] + hardcoded_coordinates[1][0]) / 2)
                center_y = int((hardcoded_coordinates[0][1] + hardcoded_coordinates[1][1]) / 2)
                # Calculate the center of the third and fourth coordinates
                center_x2 = int((hardcoded_coordinates[2][0] + hardcoded_coordinates[3][0]) / 2)
                center_y2 = int((hardcoded_coordinates[2][1] + hardcoded_coordinates[3][1]) / 2)
                
                # Draw a circle at the center of the first two coordinates
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
            
                cv2.circle(frame, (center_x2, center_y2), 5, color=(0, 0, 255), thickness=-1)
                #cv2.line(frame, (center_x2, center_y2), (center_x, center_y), (0, 0, 255), 2)

                # Calculate the coordinates 200 pixels before the center
                prev_center_x = center_x + 330        
                # Draw a circle at the previous center coordinates
                cv2.circle(frame, (prev_center_x, center_y), 5, color=(0, 0, 255), thickness=-1)
                cv2.line(frame, (prev_center_x, center_y), (center_x, center_y), (0, 0, 255), 2)
                
                # Calculate the coordinates 200 pixels before the center
                prev_center_x2 = prev_center_x + 330        
                # Draw a circle at the previous center coordinates
                cv2.circle(frame, (prev_center_x2, center_y), 5, color=(0, 0, 255), thickness=-1)
                cv2.line(frame, (prev_center_x, center_y), (center_x, center_y), (0, 0, 255), 2)
                
                #cv2.line(frame, (center_x, center_y), (center_x2, center_y2), (0, 0, 255), 2)
                # Filter ball_centers - keep only those that are inside the polygon
                #inside_ball_centers = [pt for pt in ball_centers if cv2.pointPolygonTest(polygon_pts, pt, False) >= 0]
                """
                testcontors = red_contors(frame)
                # If the red cross centers have not been computed yet
                if red_cross_centers is None:
                    red_cross_centers = []
                    for contour in testcontors:
                        if cv2.contourArea(contour) < 5:
                            continue

                        for point in contour:
                            point = tuple(point[0])  # convert from [[x y]] to (x, y)
                            if cv2.pointPolygonTest(polygon_pts, (int(point[0]), int(point[1])), False) >= 0:
                                grid_point = (point[0] // 10, point[1] // 10)  # Convert from pixel coordinates to grid coordinates
                                red_cross_centers.append(point)
                                cv2.circle(frame, point, 5, (255, 255, 255), -1)  # Draw the center of the red cross in light blue

                        #print(f"Red Cross at: {red_cross_centers}")
                """
                num = 80
                # top left corner
                left_start = (hardcoded_coordinates[0][0] + num, hardcoded_coordinates[0][1] + num)
                #top right corner
                upper_end = (hardcoded_coordinates[3][0] - num, hardcoded_coordinates[3][1] + num)
                #bottom left corner
                down_start = (hardcoded_coordinates[1][0] + num, hardcoded_coordinates[1][1] - num)
                #bottom right corner
                right_start = (hardcoded_coordinates[2][0] - num, hardcoded_coordinates[2][1] - num)
                # CORNER TO PUSH THE CROSS
                cv2.circle(frame, (upper_end), 5, (0, 255, 255), -1)
                # Bottom right corner
                #cv2.circle(frame, (hardcoded_coordinates[2][0] - 80, hardcoded_coordinates[2][1] - 80), 5, (0, 255, 255), -1)
                
                if pushstart:
                    getdistance = distance(start, (upper_end))
                    robo_angle = angle = calculate_angle(robot, pink, (prev_center_x2, center_y))
                    if not firstpushstart:
                        firstpushstart = True
                        pushstarttime = time.time()
                    if getdistance < 100 or (time.time() - pushstarttime > 10):
                        pushstart = False
                        turnback = 100
                        start_time = time.time()
                
                if time.time() - start_time > 2:
                    turnback = 10000

                if turngoalpoint >= 3:
                    turngoalpoint = 0
                    forcedgotogoal = False
                    last_update_time = time.time()
                    
                """
                # Find the shortest path to the closest ball using Dijkstra's algorithm
                greendist = cv2.pointPolygonTest(polygon_pts, tuple(start), True)
                pinkdist = cv2.pointPolygonTest(polygon_pts, tuple(pink), True)

                
                if (pinkdist >= 0 and pinkdist <= 100) and (greendist >= 0 and greendist <= 100) and greendist > pinkdist:
                    print("pink hit = ", pinkdist)
                    print("green hit = ", greendist)
                    print("pink less than green, turnback")
                    goforward = greendist
                elif (pinkdist >= 0 and pinkdist <= 100) and (greendist >= 0 and greendist <= 100) and greendist < pinkdist:
                    print("pink hit = ", pinkdist)
                    print("green hit = ", greendist)
                    print("green less than pink, goforward")
                    turnback = pinkdist
                elif not move_backward:
                    turnback = 10000
                    goforward = 10000
                """
                if not pushstart:
                    # Calculate the elapsed time in seconds
                    elapsed_time = int(time.time() - last_update_time)

                    # Format the elapsed time as minutes:seconds
                    minutes = elapsed_time // 60
                    seconds = elapsed_time % 60
                    timer_text = f"Time: {minutes:02d}:{seconds:02d}"

                    # Draw the timer text in the top-left corner of the frame
                    cv2.putText(frame, timer_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Reset the ball if it is stuck
                    if turnright > 6 or turnleft > 6:
                        print("resseting because to many turns, turnright = ", turnright, " turnleft = ", turnleft)
                        turnamt = 0
                        prepointmarked = False
                        prepoint = None
                        closest_ball = None
                        previous_ball = None
                        turnright = 0
                        turnleft = 0

                    if time.time() - last_update_time >= 300:
                        forcedgotogoal = True
                        last_update_time = time.time() + 200
                    #waypoint = [0, 0]
                    if len(ball_centers) <= 0 or forcedgotogoal:
                        #closest_ball = (prev_center_x, center_y)
                        closest_ball = {
                            'id': 'Goal',
                            'center': (prev_center_x, center_y)
                        }
                        prepointmarked = False
                        #previous_ball = closest_ball  # Update the previous_ball for the next iteration
                        #print(distance(start, closest_ball))
                        if not distancecheck:
                            robo_angle = angle = calculate_angle(robot, pink, closest_ball['center'])
                        else:
                            robo_angle = angle = calculate_angle(robot, pink, (prev_center_x2, center_y))
                        #print(robo_angle)
                        getdistance = distance(start, closest_ball['center'])
                        if getdistance < 10:
                            distancecheck = True
                            print("distance is less than 10: ", getdistance)
                            robo_angle = angle = calculate_angle(robot, pink, (prev_center_x2, center_y))
                            goal = True
                    else:
                        if not safepoint:
                            if closest_ball is None or (previous_ball is not None and previous_ball['id'] not in [ball['id'] for ball in ball_centers]):
                                #print("closest ball = ", safepoint)
                                closest_ball = selected_ball(start, ball_centers)
                    
                        previous_ball = closest_ball
                        if prepoint:
                            robo_angle = angle = calculate_angle(robot, pink, prepoint)
                            getdistance = distance(start, prepoint)
                        else:
                            robo_angle = angle = calculate_angle(robot, pink, closest_ball['center'])
                            getdistance = distance(start, closest_ball['center'])
                        goal = False
                        goal2 = False
                        distancecheck = False
                        # Check if the closest ball is close to the polygon_pts polygon

                    if closest_ball is not None and start is not None and frame is not None and pink is not None:
                        distance_to_polygon = calculate_distance_to_polygon(closest_ball['center'], polygon_pts)

                        if prepointmarked:
                            turnback = 100
                            turnamt = 0

                        if angle is not None:
                            cv2.putText(frame, "Angle: {:.2f}".format(angle), (frame_center), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)  # Draw lines from the robot to each white ball
                        if distance(start, closest_ball['center']) >= distance_threshold:
                            cv2.line(frame, start, closest_ball['center'], (255, 255, 0), 2)
                            cv2.putText(frame, "Tracking Ball {}".format(closest_ball['id']), (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 255), 2)
                            
                            if distance_to_polygon < 180 and not prepointmarked:
                                prepoint = draw_line_to_nearest_boundary(frame, closest_ball['center'], boundaries)

                                if prepoint and distance(prepoint, start) < 20:
                                    print("distance = ", distance(prepoint, closest_ball['center']))
                                    prepoint = None
                                    prepointmarked = True
                        else:
                            print("HIT ELSE***************")
                            safepoint = False
                            #start_time = time.time()
                            closest_ball = None

                            if prepointmarked:
                                print("prepointmarked = True")
                                turnback = 100
                                turnamt = 0
                                prepointmarked = False
                                prepoint = None
                                start_time = time.time()
                            
                        if time.time() - start_time >= 2:
                            turnback = 10000

                        if turnamt > 30:
                            print("going to safepoint")
                            safepoint = True
                            closest_ball = {
                                'id': 'Safepoint',
                                'center': (prev_center_x, center_y)
                            }
                            #start_time = time.time()

                        #cv2.line(frame, waypoint, closest_ball, (255, 255, 0), 2)
                        #cv2.circle(frame, waypoint, 5, (0, 0, 255), -1)  # Draw a circle at the closest ball
                        # Display the original frame with bounding boxes and lines
                cv2.imshow('Frame', frame)

                # Wait
                key = cv2.waitKey(1) & 0xFF

                # If the 'q' key is pressed, break from the loop
                if key == ord('q'):
                    break
    except Exception as e:
        traceback.print_exc()
        retry = True
    else:
        retry = False


# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()