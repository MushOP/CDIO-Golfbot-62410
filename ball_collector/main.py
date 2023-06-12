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

SERVER_IP = '172.20.10.3'  # Replace with the IP address of your server
SERVER_PORT = 5001

# Create a DriveBase object to control the motors
ev3 = EV3Brick()
left_wheel = Motor(Port.B)
right_wheel = Motor(Port.D)
ultra = UltrasonicSensor(Port.S4)
front_arm = Motor(Port.C)
back_arm = Motor(Port.A)
robot = DriveBase(left_wheel, right_wheel, wheel_diameter=55.5, axle_track=104)

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
    #json_data = json.loads(response)
    print("Parsed JSON:", command)
    if 'left' in command:
        robot.drive(0, 50)
        # print('left-angle: ', command['left'])
        #robot.turn(command['left'])
    elif 'right' in command:
        # print('right-angle: ', command['right'])
        robot.drive(0, -50)
        #robot.turn(-command['right'])
    elif 'onpoint' in command:
        print("I'm at around angle 0", command['onpoint'])
        #robot.stop()
        robot.drive(100, 0)
        left_arm = Motor(Port.C, Direction.COUNTERCLOCKWISE, [12, 36]) 
        left_arm.control.limits(speed=150, acceleration=120)
        left_arm.run(150)
    elif 'goal_point' in command:
        print("im at point")
        run = True
        robot.turn(180-command["goal_point"])
    else:
        print('Something went wrong: ', command['idk'])
    
    # Close the socket connection
    sock.close()
    
    # Delay before making the next request
    #time.sleep(1)
    
    
    # Parse the command and set the motor speeds
    if command == 'forward':
        robot.drive(200, 0)
        left_arm = Motor(Port.C, Direction.COUNTERCLOCKWISE, [12, 36])
        left_arm.control.limits(speed=100, acceleration=120)
        left_arm.run(100)
        if ultra.distance() < 400:
            print("ultra: ", ultra.distance())
            print("Turning inside if")
            s.sendall('Turning'.encode())
            wait(3)
            robot.turn(200)
        print("ultra: ", ultra.distance())
    elif command == 'backward':
        left_arm.run(0)
        robot.drive(-200, 0)
    elif command == 'left':
        robot.drive(0, 200)
    elif command == 'right':
        robot.drive(0, -200)
    elif command == 'stop':
        robot.stop()
    elif command == "open":
        port = Motor(Port.A, Direction.CLOCKWISE, [12, 36])
        port.control.limits(speed=60, acceleration=120)
        port.run(60)
    elif command == "close":
        port = Motor(Port.A, Direction.COUNTERCLOCKWISE, [12, 36])
        port.control.limits(speed=60, acceleration=120)
        port.run(60)
    while ultra.distance() < 400:                   #kode delen for at robotten kan køre plus se om der er en væg ( den som var vist på video)
        print("Turning")
        s.sendall('Turning'.encode())
        wait(4)
        robot.turn(250)