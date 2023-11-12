import socket
import threading

BUFF_SIZE = 1024
# Create a socket object
sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sender.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,BUFF_SIZE)

send_ip = ""
port = 6677

def send():
    i = 0
    for i in range(100):
        message = "Hello World"
        sender.sendto(message.encode("utf-8"), (send_ip, port))
        print("Sent: %s" % message)
    sender.close()
    print("Done")

if __name__ == '__main__':
    thread1 = threading.Thread(target=send)
    thread1.start()