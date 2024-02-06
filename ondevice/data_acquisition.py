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

import gpio_drivers as gpio_drivers

lock = mp.Lock()

_TIME_LENGTH = const(25)
_VOTE_LENGTH = const(150)


def create_shared_memory(time_length):
    done_tmp = np.array([False], dtype=bool)
    mutex_tmp = np.array([0, 0, 0], dtype=np.int8)
    data_buffer_1_tmp = np.zeros((time_length, 64), dtype=np.float32)
    data_buffer_2_tmp = np.zeros((time_length, 64), dtype=np.float32)
    data_buffer_3_tmp = np.zeros((time_length, 64), dtype=np.float32)
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
    for i in range(125000):
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

def main_process(compression_method, window_length, done_shr_name, mutex_shr_name, buff_1_shr_name, buff_2_shr_name, buff_3_shr_name):
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
    gesture_sample = 0
    gesture_repetition = 0
    first_training_sent = 1

    #Create data array
    data_array = np.empty((6,3,64,400), dtype=np.uint8)

    red_led, green_led = gpio_drivers.initialize_led()
    button = gpio_drivers.initialize_button()
    lcd = gpio_drivers.initialize_lcd()

    red_led.set_value(0)
    green_led.set_value(0)

    time_before_inference= 0
    previous_majority_vote_time = 0


    curr_vote = 0
    while done[0] == False:
        button_value = button.get_value()
        if button_value == 0:
            if training != 1:
                print("Button pressed")
            training = 1

        inference_ready = mutex[0]
        if inference_ready > 0:
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

            scaled_data = dp.compress_data(data.reshape(1,64), method=compression_method).reshape(64,1)


            if training == 1:                
                if gesture_sample < 100:
                    if gesture_sample == 0:
                        green_led.set_value(0)
                        red_led.set_value(1)
                        lcd.move_to(0,0)
                        lcd.clear()
                        lcd.putstr("Train Gest: \n%s"%(gesture_list[curr_gesture]))
                else:
                    if gesture_sample == 100:
                        red_led.set_value(0)
                        green_led.set_value(1)
                    sample_index = gesture_sample-100
                    data_array[curr_gesture, gesture_repetition, :, sample_index] = scaled_data.flatten()
                    
                
                gesture_sample += 1
                if gesture_sample == 500:
                    curr_gesture += 1
                    gesture_sample = 0
                    if curr_gesture == len(gesture_list):
                        gesture_repetition += 1
                        curr_gesture = 0
                        if gesture_repetition == 3:
                            gesture_repetition = 0
                            training = 0
                            np.savez("data_test.npz", data=data_array)
                        green_led.set_value(0)
                        red_led.set_value(0)

                time_before_inference = time.time()

            # Increment read buffer index
            lock.acquire()
            mutex[2] = (read_buffer+1)%3
            lock.release()
    lcd.move_to(0,0)
    lcd.clear()
    lcd.backlight_off()
    lcd.display_off()

def start_process(subject, session, compression_method, fine_tuned=False, on_device=False, debug=False):
    running_model_name = "emager_%s_%s_%s"%(subject, session, compression_method)
    test_session = "002" if session == "001" else "001"
    model_path = "/home/mendel/model/%s_edgetpu.tflite"%(running_model_name)

    # Create shared memory
    done_shm, mutex_shm, buff_1_shm, buff_2_shm, buff_3_shm  = create_shared_memory(_TIME_LENGTH)
    data_sampling = mp.Process(name="adc_sampling", target=sample_data, args=(_TIME_LENGTH, done_shm.name, mutex_shm.name, buff_1_shm.name, buff_2_shm.name, buff_3_shm.name, debug))
    main = mp.Process(name="main_process", target=main_process, args=(compression_method, _TIME_LENGTH, done_shm.name, mutex_shm.name, buff_1_shm.name, buff_2_shm.name, buff_3_shm.name))
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

    subject = "002"
    start_process(subject, "001", "smart", fine_tuned=False, on_device=False, debug=False)

