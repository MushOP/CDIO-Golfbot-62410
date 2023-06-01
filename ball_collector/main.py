#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
                                 InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile


# This program requires LEGO EV3 MicroPython v2.0 or higher.o
# Click "Open user guide" on the EV3 extension tab for more information.

import socket
import time
# Set the IP address and port number for the server
HOST = '172.20.10.13'  # Replace with the IP address of your computer
PORT = 12345

# Create a new socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
s.connect((HOST, PORT))

# Send data to the server
s.sendall('Starting EV3!'.encode())

timer = 2000000000
start_time = time.time()
# Create a DriveBase object to control the motors
ev3 = EV3Brick()
left_wheel = Motor(Port.B)
right_wheel = Motor(Port.D)
ultra = UltrasonicSensor(Port.S4)
front_arm = Motor(Port.C)
back_arm = Motor(Port.A)
robot = DriveBase(left_wheel,right_wheel,wheel_diameter = 55.5,axle_track = 104)


# Write your program here.
ev3.speaker.beep()
# Loop over the commands
while True:
    if(time.time() - start_time) > timer:
        print("Timer expired")
        break
    # Receive a command
    command = s.recv(1024)
    print(command)
    
    # Decode the command from bytes to string
    command = command.decode('utf-8')
    
    # Parse the command and set the motor speeds
    if command == 'forward':
        robot.drive(500, 0)
        left_arm = Motor(Port.C, Direction.COUNTERCLOCKWISE, [12, 36])
        left_arm.control.limits(speed=60, acceleration=120)
        left_arm.run(60)
        print("ultra: ", ultra.distance())
    elif command == 'backward':
        left_arm.run(0)
        robot.drive(-500, 0)
    elif command == 'left':
        robot.drive(0, 200)
    elif command == 'right':
        robot.drive(0, -200)
    elif command == 'stop':
        robot.stop()

    while ultra.distance() < 300:                   #kode delen for at robotten kan køre plus se om der er en væg ( den som var vist på video)
        print("Turning")
        s.sendall('Turning'.encode())
        wait(10)
        robot.turn(500)

# Clean up
robot.stop()
s.close()