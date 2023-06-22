#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
import time

ev3 = EV3Brick()

while True:
    battery_voltage = ev3.battery.voltage()
    battery_percentage = (battery_voltage / 9000) * 100  # Assuming 9000mV is fully charged
    print("Battery Level: {:.2f}%".format(battery_percentage))
    time.sleep(1)  # Wait for 1 second before checking again

# Check the current battery on your EV3 Brick
