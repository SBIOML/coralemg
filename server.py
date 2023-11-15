import socket
import threading
import paramiko
import os

BUFF_SIZE = 1024
# Create a socket object
server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,BUFF_SIZE)

# Get local machine name
local_ip =""
coral_ip = ""
port = 6677

# Bind to the port
server.bind((local_ip, port))

# Now wait for client connection.
print("Server started")

def receive_data():
    i = 0
    for i in range(10000):
        packet, addr = server.recvfrom(BUFF_SIZE)
        print("Received from %s: %s" % (addr, packet))
        #decode packet
        data = packet.decode("utf-8", errors="ignore")
        print(data)
    server.close()
    print("Done")

def send_file(filepath, ip):
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=ip, username="mendel")
    sftp_client = ssh_client.open_sftp()
    sftp_client.put(filepath, "/home/mendel/model/average_model_v1_edgetpu.tflite")
    sftp_client.close()
    ssh_client.close()



if __name__ == '__main__':
    # Open a text file to get the ip address of the coral
    # filepath = "/home/etienne/Documents/Universite/maitrise/recherche/CORALEMG/model/average_model_v1_edgetpu.tflite"
    # with open("coral_ip.txt", "r") as f:
    #     coral_ip = f.read().rstrip('\n')
    # send_file(filepath, coral_ip)
    thread1 = threading.Thread(target=receive_data)
    thread1.start()
