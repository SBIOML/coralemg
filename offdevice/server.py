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

def send_file(snd_filepath, dest_filepath, ip):
    '''
    Send a file to the coral

    @param snd_filepath the path to the file to send
    @param dest_filepath the path to the destination file on the coral
    @param ip the ip address of the coral
    '''

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname=ip, username="mendel")
    sftp_client = ssh_client.open_sftp()
    sftp_client.put(snd_filepath, dest_filepath)
    sftp_client.close()
    ssh_client.close()


def send_raw_data_to_coral(dataset_name, subject, session):
    snd_filepath = "dataset/raw/%s/%s_%s_raw.npz"%(dataset_name, subject, session)
    dest_filepath = "/home/mendel/dataset/%s/%s_%s_raw.npz"%(dataset_name, subject, session)

    with open("config_scripts/ip_addrs/coral_ip.txt", "r") as f:
        coral_ip = f.read().rstrip('\n')
    send_file(snd_filepath, dest_filepath, coral_ip)

def send_model_to_coral(dataset_name, model_type, subject, session, model_tuning, compression_mode, bit, ondevice=False, tuning_range=None):

    model_name = "%s_%s_%s_%s_%s_%sbits"%(dataset_name, model_type, subject, session, compression_mode, bit)
    if ondevice:
        model_name += "_ondevice"
    else:
        if tuning_range is not None:
            model_name += "_tuned_%s_%s"%(tuning_range[0], tuning_range[-1])
    
    snd_filepath = "offdevice/model/tflite/%s/%s_edgetpu.tflite"%(model_tuning,model_name)
    dest_filepath = "/home/mendel/model/%s_edgetpu.tflite"%(model_name)

    with open("config_scripts/ip_addrs/coral_ip.txt", "r") as f:
        coral_ip = f.read().rstrip('\n')
    send_file(snd_filepath, dest_filepath, coral_ip)

if __name__ == '__main__':
    #Open a text file to get the ip address of the coral

    subject = "00"
    sessions = ["1", "2"]
    compressed_methods = ["minmax", "msb", "smart", "root"]
    bits = [1,2,3,4,5,6,7,8]
    model_type = "cnn"
    dataset_name = "emager"

    for session in sessions:
        pass
        #send_raw_data_to_coral(dataset_name, subject, session)
    for compression in compressed_methods:
        for session in sessions:
            for bit in bits:
                for i in range(5):
                    tuning_range = range(i*2, i*2+2)
                    #send_model_to_coral(subject, session, "tuned", compression, ondevice=False, tuning_range=tuning_range)
                send_model_to_coral(dataset_name, model_type, subject, session, "normal", compression, bit, ondevice=False)
                #send_model_to_coral(subject, session, "extractor", compression, ondevice=True)

    
    #send_model_to_coral("002", "003", "normal", "smart", ondevice=False)
    #receive_data()

    # thread1 = threading.Thread(target=receive_data)
    # thread1.start()
