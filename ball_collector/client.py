import socket

SERVER_IP = '172.20.10.4'  # Replace with the IP address of your server
SERVER_PORT = 5001

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
client_socket.connect((SERVER_IP, SERVER_PORT))

print('connected')