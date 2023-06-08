import rpyc
import time
# Set the IP address and port number for the server
#HOST = '172.20.10.6'  # Replace with the IP address of your computer
#PORT = 12345

print(rpyc.__version__)
print('attempting to connect')
conn = rpyc.classic.connect("172.20.10.11")
print('connected')