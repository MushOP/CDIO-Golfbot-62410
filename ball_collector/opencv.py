import cv2
import numpy as np
import math
from flask import Flask, jsonify
app = Flask(__name__)
import threading
import time
import traceback
import logging
# Ignore logs from flask everytime there is a request (spams the console.)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

last_update_time = time.time()

# Camera 1 is default on windows, 0 is default on mac.
cap = cv2.VideoCapture(0)
MAX_DISTANCE = 1000000

# Loop through nodes(balls) and find the closest
def closest_node(node, nodes):
    pt = []
    dist = 9999999

    for n in nodes:
        if distance(node, n['center']) <= dist:
            dist = distance(node, n['center'])
            pt = n

    return pt

# Calculate distance between to points with their x and y
def distance(pt1, pt2):
    pt1 = np.array((pt1[0], pt1[1]))
    pt2 = np.array((pt2[0], pt2[1]))
    return np.linalg.norm(pt1 - pt2)

# Detect the green square
def detect_robot(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the HSV values for the green rectangle
    hsv_values = {'hmin': 29, 'smin': 45, 'vmin': 27, 'hmax': 92, 'smax': 255, 'vmax': 255}
    #hsv_values = {'hmin': 40, 'smin': 75, 'vmin': 20, 'hmax': 80, 'smax': 255, 'vmax': 255}
    lower_green = np.array([hsv_values['hmin'], hsv_values['smin'], hsv_values['vmin']])
    upper_green = np.array([hsv_values['hmax'], hsv_values['smax'], hsv_values['vmax']])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the HSV values for the green rectangle
    #hsv_values = {'hmin': 129, 'smin': 0, 'vmin': 97, 'hmax': 180, 'smax': 197, 'vmax': 255}
    hsv_values = {'hmin': 150, 'smin': 50, 'vmin': 50, 'hmax': 180, 'smax': 255, 'vmax': 255}
    #hsv_values = {'hmin': 150, 'smin': 100, 'vmin': 100, 'hmax': 200, 'smax': 255, 'vmax': 255}
    #hsv_values = {'hmin': 170, 'smin': 50, 'vmin': 50, 'hmax': 180, 'smax': 255, 'vmax': 255}
    lower_green = np.array([hsv_values['hmin'], hsv_values['smin'], hsv_values['vmin']])
    upper_green = np.array([hsv_values['hmax'], hsv_values['smax'], hsv_values['vmax']])

    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((10, 10), np.uint8)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)

    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the pink rectangle
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

# Returns a of table tennis balls with the specific definition in houghcircles.
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

# Detect the red cross in the middle.    
def detect_red_cross(frame, polygon_pts):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the HSV values for the red color
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)

    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Detect the red cross
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

        if aspect_ratio >= 0.4 and aspect_ratio <= 1.6 and area > 500:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            red_cross_polygon = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            break

    return red_cross_polygon

# Get the angle for the pink and green rectangle only
def calculate_robotangle(robot_position, pink_position):
    if any(val is None for val in (robot_position, pink_position)):
        return 0

    robot_vector = np.array([robot_position[0] - pink_position[0], robot_position[1] - pink_position[1]])
    
    robot_magnitude = np.linalg.norm(robot_vector)
    pink_magnitude = np.linalg.norm(pink_position)
    
    if robot_magnitude == 0 or pink_magnitude == 0:
        return 0
    
    dot_product = np.dot(robot_vector, pink_position)
    
    cross_product = np.cross(robot_vector, pink_position)
    
    cos_angle = dot_product / (robot_magnitude * pink_magnitude)
    
    # Avoid values out of range
    cos_angle = np.clip(cos_angle, -1, 1)
    
    angle_rad = np.arccos(cos_angle)
    
    angle_deg = np.degrees(angle_rad)
    
    # Return minus if it's facing the wrong direction
    if cross_product > 0:
        angle_deg = -angle_deg
    
    return angle_deg

# Get the angle for green, pink and ball.
def calculate_angle(robot_position, pink_position, closest_ball_position):
    robot_vector = np.array([robot_position[0] - pink_position[0], robot_position[1] - pink_position[1]])
    ball_vector = np.array([closest_ball_position[0] - pink_position[0], closest_ball_position[1] - pink_position[1]])
    
    dot_product = np.dot(robot_vector, ball_vector)
    
    cross_product = np.cross(robot_vector, ball_vector)
    
    robot_magnitude = np.linalg.norm(robot_vector)
    ball_magnitude = np.linalg.norm(ball_vector)
    
    if robot_magnitude == 0 or ball_magnitude == 0:
        return 0
    
    cos_angle = dot_product / (robot_magnitude * ball_magnitude)
    
    # Avoid values out of range
    cos_angle = np.clip(cos_angle, -1, 1)
    
    angle = np.degrees(np.arccos(cos_angle))
    
    # Return minus if it's facing the wrong direction
    if cross_product < 0:
        angle = -angle

    return angle

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

# Used for flask, a switch statement for each needed scenario. Handling each case is done client side (main.py)
@app.route('/', methods=['GET'])
# Sends both the angle and the distance needed.
def get_angle():
    global angle_checked
    global turnamt, turnleft, turnright, pushstart, turngoalpoint
    test = 0
    switch_cases = {
        'backward': lambda: jsonify({'backward': turnback}),
        'goal_point': lambda: jsonify({'goal_point': robo_angle}),
        'onpoint': lambda: jsonify({'onpoint': (getdistance/2) + test}),
        'right': lambda: jsonify({'right': robo_angle}),
        'left': lambda: jsonify({'left': robo_angle}),
        'forward': lambda: jsonify({'forward': goforward}),
        'forward_cross': lambda: jsonify({'forward_cross': (getdistance/2)}),
        'wait': lambda: jsonify({'wait': robo_angle})
    }

    case = 'wait'  # Default case
    goal2 = False
    if turnback <= 100 or turnbackforced <= 110:
        print("turnback!!!")
        if turnamt < 10:
            case = 'backward'
            turnamt += 1
    elif goforward <= 100:
        case = 'forward'
    elif pushstart:
        case = 'forward_cross'
    elif -0.5 <= robo_angle <= 0.5 and goal:
        print("GOOOOOOOOOAL")
        case = 'goal_point'
        turngoalpoint += 1
        turnamt = 0
    elif -1 <= robo_angle <= 1:
        print("Aligned. Moving towards the ball...")
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

    return switch_cases[case]()

# Function to run the Flask app in a separate thread on port 5001
def run_flask_app():
    app.run(host='0.0.0.0', port=5001)

flask_thread = threading.Thread(target=run_flask_app)
flask_thread.start()

global robo_angle

# Coordiantes we get from getcoord.py
hardcoded_coordinates = [(394, 93), (390, 954), (1582, 1019), (1631, 105)]
# Connect the coordinates together to create a boundary,
def draw_circles_at_coordinates(frame): 
    # Define the size
    box_size = 80

    polygon_pts = np.array(hardcoded_coordinates, dtype=np.int32)
    polygon_pts = polygon_pts.reshape((-1, 1, 2))
    cv2.polylines(frame, [polygon_pts], isClosed=True, color=(255, 0, 0), thickness=2)

    # Defines the corner boxes so that each corner can be labled as a deadzone later on
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

    return polygon_pts, small_boxes

# Draw a square in the middle, this is used for prepoint and prepointmarked. 
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

def draw_line_to_nearest_boundary(frame, ball, boundaries):
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
        
        # Blue line to the nearest point on the boundary
        cv2.line(frame, ball, nearest_point_on_line, color=(255, 0, 0), thickness=2)
        
        # Draw the boundary itself here.
        cv2.line(frame, nearest_boundary_start, nearest_boundary_end, color=(0, 255, 0), thickness=2)
        
        return nearest_point_on_line
    
    return False

# Helper funktion to get the nearest point for the ball to the green rectangle in the middle (draw_line_to_nearest_boundary).
def get_nearest_point_on_line(point, line_start, line_end, line_length=200):
    line_vec = np.array(line_end) - np.array(line_start)
    point_vec = np.array(point) - np.array(line_start)
    line_len = np.dot(line_vec, line_vec)
    factor = np.dot(point_vec, line_vec) / line_len
    nearest_point = np.array(line_start) + factor * line_vec
    # Clamp factor between 0 and 1
    factor = max(0, min(1, factor))

    nearest_point = np.array(line_start) + factor * line_vec
    return tuple(map(int, nearest_point))


# Makes us able to see if the angle is off for some reason but drawing a straight line outwards between the center of pink and green.
def draw_straight_line(image, start, end, color):
    if any(val is None for val in (start, end, image, color)):
        return
    height, width = image.shape[:2]

    x1, y1 = start
    x2, y2 = end

    if x1 == x2:
        x1 = x2 = max(0, min(x1, width - 1))
        y1 = 0
        y2 = height - 1
    else:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1 
        
        x1 = 0
        y1 = int(b)
        x2 = width - 1
        y2 = int(m * x2 + b)

    cv2.line(image, (x1, y1), (x2, y2), color, 2)
    tempangle = calculate_robotangle(start, end)
    cv2.putText(frame, "Robotangle: {:.2f}".format(tempangle), (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)

# Selects the cloest_ball by running the closest_node function and returning the result.
closest_ball = None
def selected_ball(robot, balls):
    closest_ball = closest_node(robot, balls)
    return closest_ball

# Get the distance from the square, to the closest_ball
def calculate_distance_to_polygon(ball, polygon_pts):
    return cv2.pointPolygonTest(np.array(polygon_pts), ball, True)

previous_ball = None
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
        while True:
            ret, frame = cap.read()
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            polygon_pts, small_boxes = draw_circles_at_coordinates(frame)
            #polygon_pts1 = draw_circles_at_redcross(frame)

            # Detect the green rectangle (robot)
            robot = detect_robot(frame)
            
            # Get the cross
            redcross = detect_red_cross(frame, polygon_pts)

            # Get all the table tennis balls (orange and white)
            ball_centers = getBalls(frame, polygon_pts, small_boxes, redcross)
            pink = detect_pink(frame)

            start = robot[:2] if robot is not None else None
            boundaries = draw_line_points(frame)

            # Check if the robot was found
            if start is None:
                print("Starting point not found. Adjust camera position or choose a different starting point.")
            else:
                draw_straight_line(frame, start, pink, (0, 255, 0))
                
                
                frame_height, frame_width = frame.shape[:2]
                frame_center = (frame_width // 2, frame_height // 2)

                # Mark the half of hardcoded coordinates as the goal (draw a circle)
                center_x = int((hardcoded_coordinates[0][0] + hardcoded_coordinates[1][0]) / 2)
                center_y = int((hardcoded_coordinates[0][1] + hardcoded_coordinates[1][1]) / 2)

                # Calculate the center of the third and fourth coordinates
                center_x2 = int((hardcoded_coordinates[2][0] + hardcoded_coordinates[3][0]) / 2)
                center_y2 = int((hardcoded_coordinates[2][1] + hardcoded_coordinates[3][1]) / 2)
                
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
                cv2.circle(frame, (center_x2, center_y2), 5, color=(0, 0, 255), thickness=-1)
    

                prev_center_x = center_x + 330        
                # Draw a circle at the previous center coordinates with an offset of 330
                cv2.circle(frame, (prev_center_x, center_y), 5, color=(0, 0, 255), thickness=-1)
                cv2.line(frame, (prev_center_x, center_y), (center_x, center_y), (0, 0, 255), 2)
                
                
                prev_center_x2 = prev_center_x + 330        
                cv2.circle(frame, (prev_center_x2, center_y), 5, color=(0, 0, 255), thickness=-1)
                cv2.line(frame, (prev_center_x, center_y), (center_x, center_y), (0, 0, 255), 2)
                
                # Define each corner
                num = 80
                # top left corner
                left_start = (hardcoded_coordinates[0][0] + num, hardcoded_coordinates[0][1] + num)
                #top right corner
                upper_end = (hardcoded_coordinates[3][0] - num, hardcoded_coordinates[3][1] + num)
                #bottom left corner
                down_start = (hardcoded_coordinates[1][0] + num, hardcoded_coordinates[1][1] - num)
                #bottom right corner
                right_start = (hardcoded_coordinates[2][0] - num, hardcoded_coordinates[2][1] - num)

                # CORNER TO PUSH THE CROSS (changeable for each corner)
                cv2.circle(frame, (upper_end), 5, (0, 255, 255), -1)
                
                # Fpr the first few seconds, push the square in a corner.
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
                
                # Stop going back after 2 seconds
                if time.time() - start_time > 2:
                    turnback = 10000

                # If it keeps trying to return a ball after the forcedgotogoal, stop after 3 tries.
                if turngoalpoint >= 3:
                    turngoalpoint = 0
                    forcedgotogoal = False
                    last_update_time = time.time()
                    
                if not pushstart:
                    elapsed_time = int(time.time() - last_update_time)
                    minutes = elapsed_time // 60
                    seconds = elapsed_time % 60
                    timer_text = f"Time: {minutes:02d}:{seconds:02d}"

                    # Top left corner
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

                    # If past 5 minutes, then deliver to goal no matter what
                    if time.time() - last_update_time >= 300:
                        forcedgotogoal = True
                        last_update_time = time.time() + 200

                    # When it has to deliver to goal, meaning either we find no balls or we are forced.    
                    if len(ball_centers) <= 0 or forcedgotogoal:
                        closest_ball = {
                            'id': 'Goal',
                            'center': (prev_center_x, center_y)
                        }
                        prepointmarked = False

                        if not distancecheck:
                            robo_angle = angle = calculate_angle(robot, pink, closest_ball['center'])
                        else:
                            robo_angle = angle = calculate_angle(robot, pink, (prev_center_x2, center_y))

                        getdistance = distance(start, closest_ball['center'])
                        if getdistance < 10:
                            distancecheck = True
                            print("distance is less than 10: ", getdistance)
                            robo_angle = angle = calculate_angle(robot, pink, (prev_center_x2, center_y))
                            goal = True
                    else:
                        # If we DO have balls, then get the closest_ball, if prepoint then go to that first.
                        if not safepoint:
                            if closest_ball is None or (previous_ball is not None and previous_ball['id'] not in [ball['id'] for ball in ball_centers]):
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

                    
                    if closest_ball is not None and start is not None and frame is not None and pink is not None:
                        distance_to_polygon = calculate_distance_to_polygon(closest_ball['center'], polygon_pts)

                        if prepointmarked:
                            turnback = 100
                            turnamt = 0

                        if angle is not None:
                            cv2.putText(frame, "Angle: {:.2f}".format(angle), (frame_center), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)  # Draw lines from the robot to each white ball
                        # If the robot is above 50 in range (meaning it hasn't collected it yet)
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

                cv2.imshow('Frame', frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
    except Exception as e:
        traceback.print_exc()
        retry = True
    else:
        retry = False

cap.release()
cv2.destroyAllWindows()