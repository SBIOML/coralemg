import socket
import threading

BUFF_SIZE = 1024
# Create a socket object
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,BUFF_SIZE)

# Get local machine name
ip =""
port = 6677

# Bind to the port
server.bind((ip, port))

# Now wait for client connection.
print("Server started")

def receive_data():
    i = 0
    for i in range(100):
        packet, addr = server.recvfrom(BUFF_SIZE)
        print("Received from %s: %s" % (addr, packet))
        #decode packet
        data = packet.decode("utf-8")
        print(data)
    server.close()
    print("Done")

if __name__ == '__main__':
    thread1 = threading.Thread(target=receive_data)
    thread1.start()