import time
import board
import gpiod
import numpy as np
import digitalio
import multiprocessing as mp
from multiprocessing import shared_memory
import threading
import socket
import rhd2164_driver
import data_processing as dp
from adafruit_bus_device.spi_device import SPIDevice
from micropython import const
from sklearn import preprocessing
import tpu_inference as infer

# SPI clock frequency in Hz
_SPI_FREQ_HZ         = const(4000000)
_CMD_TO_CMD_DELAY_MS = const(1000)

lock = mp.Lock()

# ADC sampling timer clock value in Hz
_ADC_SAMPLING_TIMER_CLOCK_HZ = const(1000000)
#ADC sampling timer period value
_ADC_SAMPLING_TIMER_PERIOD = const(999) #(999) -> Fs = 1 kHz
_TIME_LENGTH = const(25)
_VOTE_LENGTH = const(100)

def initialize_button():
    chip = gpiod.chip("0", gpiod.chip.OPEN_BY_NUMBER)
    button = chip.get_line(2)
    config = gpiod.line_request()
    config.request_type = gpiod.line_request.DIRECTION_INPUT
    button.request(config)
    #button.get_value()
    return button

def create_shared_memory(time_length):
    mutex_tmp = np.array([0,0], dtype=np.int8)
    data_buffer_1_tmp = np.zeros((time_length, 64), dtype=np.float32)
    data_buffer_2_tmp = np.zeros((time_length, 64), dtype=np.float32)
    # Now create a NumPy array backed by shared memory
    mutex_shm = shared_memory.SharedMemory(create=True, size=mutex_tmp.nbytes)
    buff_1_shm = shared_memory.SharedMemory(create=True, size=data_buffer_1_tmp.nbytes)
    buff_2_shm = shared_memory.SharedMemory(create=True, size=data_buffer_2_tmp.nbytes)

    mutex = np.ndarray(mutex_tmp.shape, dtype=np.int8, buffer=mutex_shm.buf)
    data_buffer_1 = np.ndarray(data_buffer_1_tmp.shape, dtype=np.float32, buffer=buff_1_shm.buf)
    data_buffer_2 = np.ndarray(data_buffer_2_tmp.shape, dtype=np.float32, buffer=buff_2_shm.buf)

    mutex[:] = mutex_tmp[:]
    data_buffer_1[:] = data_buffer_1_tmp[:]
    data_buffer_2[:] = data_buffer_2_tmp[:]

    return mutex_shm, buff_1_shm, buff_2_shm, mutex, data_buffer_1, data_buffer_2


def sample_adc(samplin_rate, window_length, mutex_shr_name, buff_1_shr_name, buff_2_shr_name):
    name = mp.current_process().name

    existing_mutex_shm = shared_memory.SharedMemory(name=mutex_shr_name)
    existing_buff1_shm = shared_memory.SharedMemory(name=buff_1_shr_name)
    existing_buff2_shm = shared_memory.SharedMemory(name=buff_2_shr_name)

    mutex = np.ndarray((2), dtype=np.int8, buffer=existing_mutex_shm.buf)
    buff1 = np.ndarray(((window_length, 64)), dtype=np.float32, buffer=existing_buff1_shm.buf)
    buff2 = np.ndarray(((window_length, 64)), dtype=np.float32, buffer=existing_buff2_shm.buf)
    print("%s, Starting"%(name))
    sampling_index = 0
    old_time = 0
    i = 0
    while True:
        current_time = time.time()
        time_diff = current_time-old_time
        if time_diff > samplin_rate:
            current_buffer = mutex[0]
            if (current_buffer == 0):
                buff1[sampling_index,:] = np.random.rand(64)
            elif (current_buffer == 1):
                buff2[sampling_index,:] = np.random.rand(64)

            if (sampling_index == window_length-1):
                lock.acquire()
                mutex[0] = (current_buffer+1)%2
                mutex[1] = 1
                lock.release()

            sampling_index = (sampling_index+1)%window_length
            old_time = current_time
    print("%s, Exiting"%(name))

def main_process(model_path, vote_length, window_length, mutex_shr_name, buff_1_shr_name, buff_2_shr_name):
    print("Configuring GPIOs /n")
    button = initialize_button()

    # Create shared memory
    existing_mutex_shm = shared_memory.SharedMemory(name=mutex_shr_name)
    existing_buff1_shm = shared_memory.SharedMemory(name=buff_1_shr_name)
    existing_buff2_shm = shared_memory.SharedMemory(name=buff_2_shr_name)

    mutex = np.ndarray((2), dtype=np.int8, buffer=existing_mutex_shm.buf)
    buff1 = np.ndarray(((window_length, 64)), dtype=np.float32, buffer=existing_buff1_shm.buf)
    buff2 = np.ndarray(((window_length, 64)), dtype=np.float32, buffer=existing_buff2_shm.buf)

    # Create server
    BUFF_SIZE = 1024
    # Get the ip address of the server
    with open("/home/mendel/ip_addr.txt", "r") as f:
        ip_addresses = f.read().rstrip('\n')
    # Get the ip address after the server string by removing the server string and get from the first number
    receiver_ip = ip_addresses[ip_addresses.find("server"):].split(" ")[-1]
    # Get the substring after the first number

    port = 6677
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,BUFF_SIZE)

    # Create interpreter
    interpreter = infer.make_interpreter(model_path)
    interpreter.allocate_tensors()

    nb_votes = int(np.floor(vote_length/window_length))
    votes_arr = np.zeros(nb_votes, dtype=np.uint8)
    curr_vote = 0
    training = 0
    state = 0
    first_run = True
    training_done = False

    label = 0
    nb_data_send = 0

    while True:
        value = button.get_value()
        data_ready = mutex[1]

        if value == 0:
            if training != 1:
                print("Button pressed")
            training = 1

        if data_ready == 1:
            current_buffer = (mutex[0]-1)%2
            if current_buffer == 0:
                data = dp.process_buffer(buff1, fs=1000, Q=30, notch_freq=60)
            elif current_buffer == 1:
                data = dp.process_buffer(buff2, fs=1000, Q=30, notch_freq=60)
            scaled_data = dp.compress_data(data, method="base").reshape(4,16,1)
            lock.acquire()
            mutex[1] = 0
            lock.release()

        if training == 1:
            curr_time = time.time()
            if state == 0:
                if data_ready == 1:
                    sender.sendto(str(label).encode("utf-8"), (receiver_ip, port))
                    #send numpy array
                    curr_time = time.time()
                    sender.sendto(scaled_data.tobytes(), (receiver_ip, port))
                    print("Time taken to send data : %f"%(time.time()-curr_time))
                    nb_data_send += 1
                if nb_data_send == 20:
                    label += 1
                    nb_data_send = 0

        elif data_ready == 1 and training == 0:
            vote = infer.make_inference(interpreter, scaled_data, True)
            votes_arr[curr_vote] = vote
            if (curr_vote+1 == nb_votes):
                majority_vote = np.argmax(np.bincount(votes_arr))
                print("Majority vote : %d"%(majority_vote))
            curr_vote = (curr_vote+1)%nb_votes


def main():
    model_path = "/home/mendel/model/average_model_v1_edgetpu.tflite"
    #print(board.board_id)
    #spi = board.SPI()
    #cs = digitalio.DigitalInOut(board.D5)

    mutex_shm, buff_1_shm, buff_2_shm, mutex, data_buffer_1, data_buffer_2 = create_shared_memory(_TIME_LENGTH)
    adc_sampling = mp.Process(name="adc_sampling", target=sample_adc, args=(0.001,_TIME_LENGTH,mutex_shm.name, buff_1_shm.name, buff_2_shm.name,))
    main_loop = mp.Process(name="main_process", target=main_process, args=(model_path, _VOTE_LENGTH, _TIME_LENGTH, mutex_shm.name, buff_1_shm.name, buff_2_shm.name,))
    adc_sampling.start()
    main_loop.start()
    adc_sampling.join()
    main_loop.join()

    print("done")
    #rhd2164 = rhd2164_driver.RHD2164(spi,cs, )

    main_loop.close()
    adc_sampling.close()

    mutex_shm.close()
    buff_1_shm.close()
    buff_2_shm.close()

    mutex_shm.unlink()
    buff_1_shm.unlink()
    buff_2_shm.unlink()

main()
