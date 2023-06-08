import cv2
import numpy as np
import math
class FrameTransformer:
    manMode = False
    selectCount = 0
    # Field dimensions
    width = 750
    height = 500
    corners = [[0, 0], [750, 0], [0, 500], [750, 500]]
    goals = [[0, 250], [750, 250]]

    def __init__(self):
        pass

    def transform(self, frame, frameCount):
        if (not self.manMode and frameCount % 20 == 0):
            self.corners = self.getCorners(frame)
            self.goals = self.getGoals(self.corners)

        if (self.corners != None):
            # Coordinates for the corners
            oldCoordinates = np.float32([self.corners[0], self.corners[1],
                                         self.corners[2], self.corners[3]])

            # New coordinates for the corners
            newCoordinates = np.float32([[0, 0], [self.width, 0],
                                         [0, self.height], [self.width, self.height]])

            # Transform image
            matrix = cv2.getPerspectiveTransform(
                oldCoordinates, newCoordinates)
            transformed = cv2.warpPerspective(
                frame, matrix, (self.width, self.height))
            # Create circles to indicate detected corners and goals
            frame = cv2.circle(frame, tuple(
                self.corners[0]), 20, (255, 0, 0), 2)
            frame = cv2.circle(frame, tuple(
                self.corners[1]), 20, (255, 0, 0), 2)
            frame = cv2.circle(frame, tuple(
                self.corners[2]), 20, (255, 0, 0), 2)
            frame = cv2.circle(frame, tuple(
                self.corners[3]), 20, (255, 0, 0), 2)
            frame = cv2.circle(frame, tuple(self.goals[0]), 20, (255, 0, 0), 2)
            frame = cv2.circle(frame, tuple(self.goals[1]), 20, (255, 0, 0), 2)
            return transformed

    def get_point(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONUP:

            self.selectCount += 1
            if (self.selectCount == 1):
                self.corners[0] = [x, y]
                print("Upper left(", x, ",", y, ")")
            elif (self.selectCount == 2):
                self.corners[1] = [x, y]
                print("Upper right(", x, ",", y, ")")
            elif (self.selectCount == 3):
                self.corners[2] = [x, y]
                print("Lower left(", x, ",", y, ")")
            elif (self.selectCount == 4):
                self.corners[3] = [x, y]
                print("Lower right(", x, ",", y, ")")
            elif (self.selectCount == 5):
                self.goals[0] = [x, y]
                print("Left goal(", x, ",", y, ")")
            elif (self.selectCount == 6):
                self.goals[1] = [x, y]
                print("Right goal(", x, ",", y, ")")
                self.selectCount = 0

    def getCorners(self, frame):
        upper_left = [99999, 99999]
        upper_right = [0, 99999]
        lower_left = [99999, 0]
        lower_right = [0, 0]

        # Define lower/upper color field for red
        lower_red = np.array([0, 0, 200], dtype="uint8")
        upper_red = np.array([100, 100, 255], dtype="uint8")

        # Frame dimensions
        height = frame.shape[0]
        width = frame.shape[1]

        redMask = cv2.inRange(frame, lower_red, upper_red)

        # Filter out everything that is not red
        wall = frame.copy()
        wall[np.where(redMask == 0)] = 0
        # wall = cv2.GaussianBlur(wall,(41,41) ,5)
        # Convert to grey
        wall = cv2.cvtColor(wall, cv2.COLOR_BGR2GRAY)

        # Find corners
        # dest = cv2.cornerHarris(np.float32(wall), 20, 3, 0.2496)
        dest = cv2.cornerHarris(np.float32(wall), 20, 3, 0.14)
        dest = cv2.dilate(dest, None)

        # frame[dest > 0.01 * dest.max()]=[0, 0, 255]

        # Iterate through the detected corners, and use the appropiate ones
        # TODO should be optimized in the future
        thresh = 0.1*dest.max()
        # if lower_right[0] == 0 :
        for j in range(0, dest.shape[0]):
            for i in range(0, dest.shape[1]):
                if (dest[j, i] > thresh):
                    if i < upper_left[0] and i < width/2 and j < upper_left[1] and j < height/2:
                        upper_left[0] = i
                        upper_left[1] = j
                    if i > upper_right[0] and i > width/2 and j < upper_right[1] and j < height/2:
                        upper_right[0] = i
                        upper_right[1] = j
                    if i < lower_left[0] and i < width/2 and j > lower_left[1] and j > height/2:
                        lower_left[0] = i
                        lower_left[1] = j
                    if i > lower_right[0] and i > width/2 and j > lower_right[1] and j > height/2:
                        lower_right[0] = i
                        lower_right[1] = j
        return [upper_left, upper_right, lower_left, lower_right]

    def getGoals(self, corners):
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

class Ball:

    def __init__(self, x, y, radius, index):
        self.x = x
        self.y = y
        self.radius = radius
        self.index = index

class Robot:

    def __init__(self, x, y, height, width):
        self.x = x
        self.y = y
        self.height = height
        self.width = width

def detectOrangeBall(frame, robot):
    orange_ball_hsv_values = {'hmin': 16, 'smin': 98, 'vmin': 191, 'hmax': 21, 'smax': 255, 'vmax': 255}
    hmin, smin, vmin = orange_ball_hsv_values['hmin'], orange_ball_hsv_values['smin'], orange_ball_hsv_values['vmin']
    hmax, smax, vmax = orange_ball_hsv_values['hmax'], orange_ball_hsv_values['smax'], orange_ball_hsv_values['vmax']
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_range = np.array([hmin, smin, vmin])
    upper_range = np.array([hmax, smax, vmax])
    mask = cv2.inRange(hsv, lower_range, upper_range)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                (x, y, w, h) = cv2.boundingRect(contour)
                radius = int(w / 2)
                if radius > 9 or radius < 7:
                    continue
                if robot and x > robot.x and x < robot.x + robot.width and y > robot.y and y < robot.y + robot.height:
                    continue
                cv2.circle(frame, (int(x + w / 2), int(y + h / 2)), int(max(w, h) / 2), (0, 255, 0), 2)
                return Ball(x + radius / 2, y + radius / 2, radius, 0)
    return None

def detectBalls(frame, robot):
    balls = []
    hsv_values = {'hmin': 0, 'smin': 0, 'vmin': 218, 'hmax': 179, 'smax': 67, 'vmax': 255}
    hmin, smin, vmin = hsv_values['hmin'], hsv_values['smin'], hsv_values['vmin']
    hmax, smax, vmax = hsv_values['hmax'], hsv_values['smax'], hsv_values['vmax']
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_range = np.array([hmin, smin, vmin])
    upper_range = np.array([hmax, smax, vmax])
    mask = cv2.inRange(hsv, lower_range, upper_range)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:
                (x, y, w, h) = cv2.boundingRect(contour)
                radius = int(w / 2)
                if radius > 15 or radius < 14:
                    continue
                #if robot and x > robot.x and x < robot.x + robot.width and y > robot.y and y < robot.y + robot.height:
                    #continue
                cv2.circle(frame, (int(x + w / 2), int(y + h / 2)), int(max(w, h) / 2), (0, 255, 0), 2)
                balls.append(Ball(x + radius / 2, y + radius / 2, radius, len(balls)))
    return balls

def detectRobot(frame):
    hsv_values = {'hmin': 70, 'smin': 38, 'vmin': 0, 'hmax': 92, 'smax': 255, 'vmax': 255}
    hmin, smin, vmin = hsv_values['hmin'], hsv_values['smin'], hsv_values['vmin']
    hmax, smax, vmax = hsv_values['hmax'], hsv_values['smax'], hsv_values['vmax']
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_range = np.array([hmin, smin, vmin])
    upper_range = np.array([hmax, smax, vmax])
    mask = cv2.inRange(hsv, lower_range, upper_range)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 3)
        center = np.mean(box, axis=0)
        robotH = math.sqrt((box[0][0] - box[1][0]) ** 2 + (box[0][1] - box[1][1]) ** 2)
        robotW = math.sqrt((box[1][0] - box[2][0]) ** 2 + (box[1][1] - box[2][1]) ** 2)
        return Robot(center[0], center[1], robotH, robotW)

def detectBlueFrame(frame):
    hsv_values = {'hmin': 91, 'smin': 171, 'vmin': 141, 'hmax': 126, 'smax': 255, 'vmax': 255}
    hmin, smin, vmin = hsv_values['hmin'], hsv_values['smin'], hsv_values['vmin']
    hmax, smax, vmax = hsv_values['hmax'], hsv_values['smax'], hsv_values['vmax']
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_range = np.array([hmin, smin, vmin])
    upper_range = np.array([hmax, smax, vmax])
    mask = cv2.inRange(hsv, lower_range, upper_range)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 3)
        center = np.mean(box, axis=0)
        return center

def drawLine(frame, x1, y1, x2, y2, robot, ball, blueframe):
    x1, y1 = int(x1), int(y1)
    x2, y2 = int(x2), int(y2)
    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    midpoint = ((x1 + x2) // 2, (y1 + y2) // 2 - 50)
    cv2.putText(frame, "Angle: {:.2f}".format(getAngle(robot, ball, blueframe)), midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "Distance: {:.2f} cm".format(getDistance(x1, y1, x2, y2)), (midpoint[0], midpoint[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def getDistance(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    distance_in_pixels = int(math.sqrt(dx * dx + dy * dy))
    return distance_in_pixels / 4.2

def getAngle(robot, ball, blueframe):
    a = [robot.x - blueframe[0], robot.y - blueframe[1]]
    b = [robot.x - ball.x, robot.y - ball.y]
    angle = np.math.atan2(np.linalg.det([a, b]), np.dot(a, b))
    angle = np.degrees(angle)
    return int(angle)

cap = cv2.VideoCapture(0)
ft = FrameTransformer()
frameCount = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    #transformed = ft.transform(frame, frameCount)
    #transformed = frame if transformed is None else transformed

    robot = detectRobot(frame)
    balls = detectBalls(frame, robot)
    blueframe = detectBlueFrame(frame)

    if robot is not None and blueframe is not None and len(balls) > 0:
        print(" " + robot, " " + blueframe, " " + len(balls))
        for ball in balls:
            drawLine(frame, robot.x, robot.y, ball.x, ball.y, robot, ball, blueframe)

    # if balls and robot:
    #     closestBall = balls[0]
    #     closestDistance = getDistance(robot.x, robot.y, closestBall.x, closestBall.y)
    #     for ball in balls:
    #         distance = getDistance(robot.x, robot.y, ball.x, ball.y)
    #         if distance < closestDistance:
    #             closestBall = ball
    #             closestDistance = distance
    #         #if blueframe is not None:
    #             drawLine(frame, robot.x, robot.y, closestBall.x, closestBall.y, ball=closestBall, robot=robot, blueframe=blueframe)

    #cv2.imshow("Transformed", transformed)
    cv2.imshow("Board", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
