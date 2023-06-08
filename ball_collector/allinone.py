#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
                                 InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile
import socket
import time
import cv2
import numpy as np
import heapq
import math

# Set the IP address and port number for the server
HOST = '172.20.10.13'  # Replace with the IP address of your computer
PORT = 1234

# Create a new socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
s.connect((HOST, PORT))

# Send data to the server
s.sendall('Starting EV3!'.encode())

timer = 2000000000
start_time = time.time()
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
