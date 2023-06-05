import socket
import time

# create socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# specify the IP address and port number of the server
host = ''
port = 1234

# bind the socket to the IP address and port number
s.bind((host, port))

# set the server to listen for incoming connections
s.listen()

print('Waiting for connection...')

while True:
    # accept the connection
    conn, addr = s.accept()
    print('Connected by', addr)

    # # send a message to the client
    # message = 'forward'
    # conn.send(message.encode())
    # print('Command: ', message)

    # time.sleep(2)
    # message = 'left'
    # conn.send(message.encode())
    # print('Command: ', message)

    # time.sleep(2)
    # message = 'right'
    # conn.send(message.encode())
    # print('Command: ', message)
    
    # time.sleep(2)
    # message = 'backward'
    # conn.send(message.encode())
    # print('Command: ', message)
    while True:
        message = input()
        if message.lower() == 'exit':
            break

        conn.send(message.encode())
        print('Sent: : ', message)
        #time.sleep(2)
