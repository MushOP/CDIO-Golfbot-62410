#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
                                 InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile
import json
import socket
import time

# This program requires LEGO EV3 MicroPython v2.0 or higher.
# Click "Open user guide" on the EV3 extension tab for more information.

SERVER_IP = '192.168.1.117'  # Replace with the IP address of your server
SERVER_PORT = 5001
# Create a DriveBase object to control the motors
ev3 = EV3Brick()
left_wheel = Motor(Port.B) 
right_wheel = Motor(Port.D)
front_arm = Motor(Port.C)
robot = DriveBase(left_wheel, right_wheel, wheel_diameter=65, axle_track=230)

# Write your program here.
ev3.speaker.beep()
run = False
while True:
    # Create a socket object
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Connect to the server
    sock.connect((SERVER_IP, SERVER_PORT))
    
    # Send an HTTP request to the server
    request = "GET / HTTP/1.1\r\nHost: {}\r\n\r\n".format(SERVER_IP)
    sock.send(request.encode())
    
    response = ""
    while True:
        data = sock.recv(1024)
        #print("data = ", data)
        if not data:
            break
        response += data.decode()

    header, _, body = response.partition("\r\n\r\n")
    command = json.loads(body)  # Moved JSON decoding here
    
    # Assuming the response is in JSON format, you can parse it
    print("Parsed JSON:", command)
    if 'wait' in command:
        robot.stop()
    elif 'forward' in command:
        dist = round(command['forward'])
        robot.straight(dist)
    elif 'backward' in command:
        robot.drive(-250,0)
    elif 'left' in command:
        robot.turn(command['left'])
    elif 'right' in command:
        robot.turn(command['right'])
    elif 'onpoint' in command:
        print("I'm at around angle 0", round(command['onpoint']))
        dist = round(command['onpoint'])
        robot.straight(dist)
        left_arm = Motor(Port.C, Direction.COUNTERCLOCKWISE, [12, 36]) 
        left_arm.control.limits(speed=150, acceleration=120)
        left_arm.run(70)
    elif 'goal_point' in command:
        print("im at point")
        run = True
        robot.straight(command['goal_point'])
        robot.stop()
        robot.drive(-100,0)
        time.sleep(2)
        back_arm = Motor(Port.A, Direction.CLOCKWISE, [12, 36])
        back_arm.control.limits(speed=60, acceleration=120)
        back_arm.run(60)
        robot.drive(250,0)
        time.sleep(2)
        back_arm = Motor(Port.A, Direction.COUNTERCLOCKWISE, [12, 36])
        back_arm.control.limits(speed=60, acceleration=120)
        back_arm.run(60)
    elif 'forward_cross' in command:
        robot.drive(280,0)
    else:
        print('Something went wrong: ', command['idk'])
    
    # Close the socket connection
    sock.close()
    
