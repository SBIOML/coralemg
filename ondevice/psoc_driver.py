import numpy as np
import serial
import time
import struct 

class PSOC(object):
    '''
    Sensor object for data logging from HD EMG sensor
    '''

    def __init__(self, serialpath, BR):
        '''
        Initialize HDSensor object, open serial communication to specified port using PySerial API
        :param serialpath: (str) - Path to serial port
        :param BR: (int) - Com port baudrate
        '''
        self.ser = serial.Serial(serialpath, BR, timeout=1)
        self.close()

        self.packet_size = 128
        self.ones_mask = np.ones(64, dtype=np.uint8)
        self.channel_map = [10, 22, 12, 24, 13, 26, 7, 28, 1, 30, 59, 32, 53, 34, 48, 36] + \
                          [62, 16, 14, 21, 11, 27, 5, 33, 63, 39, 57, 45, 51, 44, 50, 40] + \
                          [8, 18, 15, 19, 9, 25, 3, 31, 61, 37, 55, 43, 49, 46, 52, 38] + \
                          [6, 20, 4, 17, 2, 23, 0, 29, 60, 35, 58, 41, 56, 47, 54, 42]
        self.fmt = '>64h'
    def clear_buffer(self):
        '''
        Clear the serial port input buffer.
        :return: None
        '''
        self.ser.reset_input_buffer()
        return

    def close(self):
        self.ser.close()
        return

    def open(self):
        self.ser.open()
        return
    
    def read(self, bytes_to_read):
        return self.ser.read(bytes_to_read)

    def process_packet(self, data_packet, number_of_packet):
        valid_packets = []
        for i in range(number_of_packet):
            data_slice = data_packet[i*128:(i+1)*128]
            data_lsb = np.bitwise_and(data_slice[1::2], self.ones_mask)
            zero_indices = np.where(data_lsb == 0)[0]
            if len(zero_indices) == 1:
                offset = (2*zero_indices[0]+1)-1
                # Second LSB bytes
                valid_packets.append(np.roll(data_slice, -offset))
        return valid_packets

    
    def sample(self):
        curr_time = time.time()
        bytes_available = self.ser.inWaiting()
        bytes_to_read = bytes_available - (bytes_available % self.packet_size)
        samples_list = []
        if bytes_to_read > 0:
            raw_data_packet = self.read(bytes_to_read)
            data_packet = np.array(list(raw_data_packet), dtype=np.uint8)
            number_of_packet = int(len(data_packet)/128)
            processed_packets = self.process_packet(data_packet, number_of_packet)
            for packet in processed_packets:
                samples = np.asarray(struct.unpack(self.fmt, packet), dtype=np.int16)[self.channel_map]
                samples_list.append(samples)
        return np.array(samples_list)


def test_acquisition():
    device = PSOC('/dev/ttyS0', 1500000)
    device.open()
    device.clear_buffer()
    while(1):
        sample_list = device.sample()
            


if __name__ == "__main__":
    test_acquisition()