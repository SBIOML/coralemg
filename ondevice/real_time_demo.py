import time
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
import threading
import socket
import data_processing as dp
from micropython import const
import tpu_inference as infer
import sklearn.metrics as metrics
import serial
import struct
from pycoral.adapters import common
import gpio_drivers as gpio_drivers

lock = mp.Lock()

_TIME_LENGTH = const(25)
_VOTE_LENGTH = const(150)

#TODO add nb channels
def create_shared_memory(time_length, nb_channels):
    done_tmp = np.array([False], dtype=bool)
    mutex_tmp = np.array([0, 0, 0], dtype=np.int8)
    data_buffer_1_tmp = np.zeros((time_length, nb_channels), dtype=np.float32)
    data_buffer_2_tmp = np.zeros((time_length, nb_channels), dtype=np.float32)
    data_buffer_3_tmp = np.zeros((time_length, nb_channels), dtype=np.float32)
    # Now create a NumPy array backed by shared memory
    done_shm = shared_memory.SharedMemory(create=True, size=done_tmp.nbytes)
    mutex_shm = shared_memory.SharedMemory(create=True, size=mutex_tmp.nbytes)
    buff_1_shm = shared_memory.SharedMemory(create=True, size=data_buffer_1_tmp.nbytes)
    buff_2_shm = shared_memory.SharedMemory(create=True, size=data_buffer_2_tmp.nbytes)
    buff_3_shm = shared_memory.SharedMemory(create=True, size=data_buffer_3_tmp.nbytes)

    return done_shm, mutex_shm, buff_1_shm, buff_2_shm, buff_3_shm

def sample_data(window_length, done_shr_name, mutex_shr_name, buff_1_shr_name, buff_2_shr_name, buff_3_shr_name, debug=False):
    name = mp.current_process().name

    existing_done_shm = shared_memory.SharedMemory(name=done_shr_name)
    existing_mutex_shm = shared_memory.SharedMemory(name=mutex_shr_name)
    existing_buff1_shm = shared_memory.SharedMemory(name=buff_1_shr_name)
    existing_buff2_shm = shared_memory.SharedMemory(name=buff_2_shr_name)
    existing_buff3_shm = shared_memory.SharedMemory(name=buff_3_shr_name)

    done = np.ndarray((1), dtype=bool, buffer=existing_done_shm.buf)
    mutex = np.ndarray((3), dtype=np.int8, buffer=existing_mutex_shm.buf)
    buff1 = np.ndarray(((window_length, 64)), dtype=np.float32, buffer=existing_buff1_shm.buf)
    buff2 = np.ndarray(((window_length, 64)), dtype=np.float32, buffer=existing_buff2_shm.buf)
    buff3 = np.ndarray(((window_length, 64)), dtype=np.float32, buffer=existing_buff3_shm.buf)
    if debug:
        print("%s, Starting"%(name))
    sampling_index = 0
    inference_index = 0
    updated_data = 0

    ser = serial.Serial('/dev/ttyS0', 1500000, timeout=1)
    ser.close()
    ser.open()
    ser.reset_input_buffer()
    for i in range(90000):
        bytes_available = ser.inWaiting()
        bytesToRead = bytes_available - (bytes_available % 128)
        if bytesToRead > 0:
            curr_time = time.time()
            raw_data_packet = ser.read(bytesToRead)
            data_packet = np.array(list(raw_data_packet), dtype=np.uint8)
            reordered_packet = gpio_drivers.reorder(data_packet, gpio_drivers.mask, 63)
            processed_packets = []
            fmt = '>64h'
            if reordered_packet is not None:
                for packet in reordered_packet:
                    samples = np.array(struct.unpack(fmt, packet), dtype=np.int16)
                    processed_packets.append(samples[gpio_drivers.channel_map])
                updated_data = 1

        if updated_data:
            updated_data = 0
            for data in processed_packets:
                write_buffer = mutex[1]
                if (write_buffer == 0):
                    buff1[sampling_index,:] = data
                elif (write_buffer == 1):
                    buff2[sampling_index,:] = data
                elif (write_buffer == 2):
                    buff3[sampling_index,:] = data

                if (sampling_index == window_length-1):
                    inference_index += 1
                    lock.acquire()
                    mutex[0] += 1
                    mutex[1] = (write_buffer+1)%3
                    lock.release()

                sampling_index = (sampling_index+1)%window_length
    ser.close()
    done[0] = True

def main_process(model_path, compression_method, residual_bits, vote_length, window_length, done_shr_name, mutex_shr_name, buff_1_shr_name, buff_2_shr_name, buff_3_shr_name, debug=False):
    # Create shared memory
    existing_done_shm = shared_memory.SharedMemory(name=done_shr_name)
    existing_mutex_shm = shared_memory.SharedMemory(name=mutex_shr_name)
    existing_buff1_shm = shared_memory.SharedMemory(name=buff_1_shr_name)
    existing_buff2_shm = shared_memory.SharedMemory(name=buff_2_shr_name)
    existing_buff3_shm = shared_memory.SharedMemory(name=buff_3_shr_name)

    done = np.ndarray((1), dtype=bool, buffer=existing_done_shm.buf)
    mutex = np.ndarray((3), dtype=np.int8, buffer=existing_mutex_shm.buf)
    buff1 = np.ndarray(((window_length, 64)), dtype=np.float32, buffer=existing_buff1_shm.buf)
    buff2 = np.ndarray(((window_length, 64)), dtype=np.float32, buffer=existing_buff2_shm.buf)
    buff3 = np.ndarray(((window_length, 64)), dtype=np.float32, buffer=existing_buff3_shm.buf)

    gesture_list = ["Fist","Thumb","Grip","Neutral","Pinch","Index"]

    #Training related
    training = 0
    curr_gesture = 0
    sent_gesture = 0
    first_training_sent = 1

    # Create server
    BUFF_SIZE = 1024
    # Get the ip address of the server
    with open("/home/mendel/ip_addr.txt", "r") as f:
        ip_addresses = f.read().rstrip('\n')
    # Get the ip address after the server string by removing the server string and get from the first number
    receiver_ip = ip_addresses[ip_addresses.find("server"):].split(" ")[-1]
    # Get the substring after the first num
    port = 6677
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF,BUFF_SIZE)

    red_led, green_led = gpio_drivers.initialize_led()
    button = gpio_drivers.initialize_button()
    lcd = gpio_drivers.initialize_lcd()

    red_led.set_value(0)
    green_led.set_value(0)

    time_before_inference= 0
    previous_majority_vote_time = 0

    # Create interpreter
    interpreter = infer.make_interpreter(model_path)
    #TODO check if correct
    width, height = common.input_size(interpreter)
    interpreter.allocate_tensors()
    infer.make_inference(interpreter, np.random.rand(64).reshape(width,height,1))

    nb_votes = int(np.floor(vote_length/window_length))
    votes_arr = np.zeros(nb_votes, dtype=np.uint8)
    curr_vote = 0
    while done[0] == False:
        button_value = button.get_value()
        if button_value == 0:
            if training != 1:
                print("Button pressed")
            training = 1

        inference_ready = mutex[0]
        if inference_ready > 0:
            processing_start = time.perf_counter()
            if debug:
                print("Inference ready :", inference_ready)
                time_inf = time.time()-time_before_inference
                print("Time before inference: %.1fms"%(time_inf*1000))

            lock.acquire()
            mutex[0] -= 1
            lock.release()
            read_buffer = mutex[2]
            if read_buffer == 0:
                data = dp.process_buffer(buff1, fs=1000, Q=30, notch_freq=60)
            elif read_buffer == 1:
                data = dp.process_buffer(buff2, fs=1000, Q=30, notch_freq=60)
            elif read_buffer == 2:
                data = dp.process_buffer(buff3, fs=1000, Q=30, notch_freq=60)

            scaled_data = dp.compress_data(data.reshape(1,64), compression_method, residual_bits).reshape(width,height,1)

            if training == 1:                
                if sent_gesture < 100:
                    if sent_gesture == 0:
                        green_led.set_value(0)
                        red_led.set_value(1)
                        lcd.move_to(0,0)
                        lcd.clear()
                        lcd.putstr("Train Gest: \n%s"%(gesture_list[curr_gesture]))
                else:
                    if sent_gesture == 101:
                        red_led.set_value(0)
                        green_led.set_value(1)
                    sender.sendto(str(curr_gesture).encode("utf-8"), (receiver_ip, port))
                    sender.sendto(scaled_data.tobytes(), (receiver_ip, port))
                
                sent_gesture += 1
                if sent_gesture == 500:
                    curr_gesture += 1
                    sent_gesture = 0
                    if curr_gesture == len(gesture_list):
                        curr_gesture = 0
                        training = 0
                        green_led.set_value(0)
                        red_led.set_value(0)

            else:
                vote = infer.make_inference(interpreter, scaled_data, False)
                votes_arr[curr_vote] = vote

                if (curr_vote+1 == nb_votes):
                    current_majority_vote_time = time.time()
                    time_before_majority_vote = current_majority_vote_time-previous_majority_vote_time
                    previous_majority_vote_time = current_majority_vote_time
                    majority_vote = np.argmax(np.bincount(votes_arr))
                    lcd.clear()
                    lcd.move_to(0,0)
                    lcd.putstr("Gesture: %s"%(gesture_list[majority_vote]))

                    if debug:
                        print("Time before majority vote: %.1fms"%(time_before_majority_vote*1000))
                        print("Majority vote: %s"%(majority_vote))

                curr_vote = (curr_vote+1)%nb_votes
                time_before_inference = time.time()

            # Increment read buffer index
            lock.acquire()
            mutex[2] = (read_buffer+1)%3
            lock.release()
    lcd.move_to(0,0)
    lcd.clear()
    lcd.backlight_off()
    lcd.display_off()

def start_process(dataset, subject, session, compression_method, residual_bits, fine_tuned=False, on_device=False, debug=False):
    running_model_name = "%s_%s_%s_%s_%sbits"%(dataset, subject, session, compression_method, residual_bits)
    test_session = "002" if session == "001" else "001"
    model_path = "/home/mendel/model/%s_edgetpu.tflite"%(running_model_name)

    # Create shared memory
    done_shm, mutex_shm, buff_1_shm, buff_2_shm, buff_3_shm  = create_shared_memory(_TIME_LENGTH)
    data_sampling = mp.Process(name="adc_sampling", target=sample_data, args=(_TIME_LENGTH, done_shm.name, mutex_shm.name, buff_1_shm.name, buff_2_shm.name, buff_3_shm.name, debug))
    main = mp.Process(name="main_process", target=main_process, args=(model_path, compression_method, residual_bits, _VOTE_LENGTH, _TIME_LENGTH, done_shm.name, mutex_shm.name, buff_1_shm.name, buff_2_shm.name, buff_3_shm.name, debug))
    main.start()
    time.sleep(10)
    data_sampling.start()

    data_sampling.join()
    main.join()

    print("done")
    main.close()
    data_sampling.close()

    done_shm.close()
    mutex_shm.close()
    buff_1_shm.close()
    buff_2_shm.close()
    buff_3_shm.close()

    done_shm.unlink()
    mutex_shm.unlink()
    buff_1_shm.unlink()
    buff_2_shm.unlink()
    buff_3_shm.unlink()


if __name__ == '__main__':
    dataset = "emager"
    subject = "002"
    residual_bits = 8
    start_process(dataset, subject, "001", "root", residual_bits, fine_tuned=False, on_device=False, debug=False)

