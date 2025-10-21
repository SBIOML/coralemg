import time
import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory
import threading
import socket
import data_processing as dp
import dataset_definition as dtdef
from micropython import const
import tpu_inference as infer
import sklearn.metrics as metrics
from pycoral.adapters import common
lock = mp.Lock()

_TIME_LENGTH = const(25)
_VOTE_LENGTH = const(150)


def create_sampled_data(raw_dataset_path, data_range):
    with np.load(raw_dataset_path) as data:
        raw_data = data['data']

    raw_data = raw_data[:,data_range,:,:]
    X, y = dp.extract_with_labels(raw_data)
    y = np.array(y, dtype=np.uint8)
    return X, y

def create_shared_memory(time_length, nb_sensors):
    done_tmp = np.array([False], dtype=bool)
    mutex_tmp = np.array([0, 0, 0], dtype=np.int8)
    data_buffer_1_tmp = np.zeros((time_length, nb_sensors), dtype=np.float32)
    data_buffer_2_tmp = np.zeros((time_length, nb_sensors), dtype=np.float32)
    data_buffer_3_tmp = np.zeros((time_length, nb_sensors), dtype=np.float32)
    # Now create a NumPy array backed by shared memory
    done_shm = shared_memory.SharedMemory(create=True, size=done_tmp.nbytes)
    mutex_shm = shared_memory.SharedMemory(create=True, size=mutex_tmp.nbytes)
    buff_1_shm = shared_memory.SharedMemory(create=True, size=data_buffer_1_tmp.nbytes)
    buff_2_shm = shared_memory.SharedMemory(create=True, size=data_buffer_2_tmp.nbytes)
    buff_3_shm = shared_memory.SharedMemory(create=True, size=data_buffer_3_tmp.nbytes)

    return done_shm, mutex_shm, buff_1_shm, buff_2_shm, buff_3_shm

def sample_data(data, nb_sensors, samplin_rate, window_length, done_shr_name, mutex_shr_name, buff_1_shr_name, buff_2_shr_name, buff_3_shr_name, debug=False):
    name = mp.current_process().name

    existing_done_shm = shared_memory.SharedMemory(name=done_shr_name)
    existing_mutex_shm = shared_memory.SharedMemory(name=mutex_shr_name)
    existing_buff1_shm = shared_memory.SharedMemory(name=buff_1_shr_name)
    existing_buff2_shm = shared_memory.SharedMemory(name=buff_2_shr_name)
    existing_buff3_shm = shared_memory.SharedMemory(name=buff_3_shr_name)

    done = np.ndarray((1), dtype=bool, buffer=existing_done_shm.buf)
    mutex = np.ndarray((3), dtype=np.int8, buffer=existing_mutex_shm.buf)
    buff1 = np.ndarray(((window_length, nb_sensors)), dtype=np.float32, buffer=existing_buff1_shm.buf)
    buff2 = np.ndarray(((window_length, nb_sensors)), dtype=np.float32, buffer=existing_buff2_shm.buf)
    buff3 = np.ndarray(((window_length, nb_sensors)), dtype=np.float32, buffer=existing_buff3_shm.buf)
    if debug:
        print("%s, Starting"%(name))
    sampling_index = 0
    old_time = 0
    inference_index = 0
    curr_data_index = 0
    nb_data_to_sample = len(data)
    while curr_data_index < nb_data_to_sample:
        current_time = time.time()
        time_diff = current_time-old_time
        if time_diff >= samplin_rate:
            write_buffer = mutex[1]
            if (write_buffer == 0):
                buff1[sampling_index,:] = data[curr_data_index,:]
            elif (write_buffer == 1):
                buff2[sampling_index,:] = data[curr_data_index,:]
            elif (write_buffer == 2):
                buff3[sampling_index,:] = data[curr_data_index,:]

            if (sampling_index == window_length-1):
                if debug:
                    print("Mean sampling : %s"%(inference_index))
                inference_index += 1
                lock.acquire()
                mutex[0] += 1
                mutex[1] = (write_buffer+1)%3
                lock.release()

            curr_data_index += 1
            sampling_index = (sampling_index+1)%window_length
            old_time = current_time
    
    done[0] = True

def inference_process(queue, model_path, nb_sensors, filtering_utility, compression_method, residual_bits, vote_length, window_length, done_shr_name, mutex_shr_name, buff_1_shr_name, buff_2_shr_name, buff_3_shr_name, debug=False):
    # Create shared memory
    existing_done_shm = shared_memory.SharedMemory(name=done_shr_name)
    existing_mutex_shm = shared_memory.SharedMemory(name=mutex_shr_name)
    existing_buff1_shm = shared_memory.SharedMemory(name=buff_1_shr_name)
    existing_buff2_shm = shared_memory.SharedMemory(name=buff_2_shr_name)
    existing_buff3_shm = shared_memory.SharedMemory(name=buff_3_shr_name)

    done = np.ndarray((1), dtype=bool, buffer=existing_done_shm.buf)
    mutex = np.ndarray((3), dtype=np.int8, buffer=existing_mutex_shm.buf)
    buff1 = np.ndarray(((window_length, nb_sensors)), dtype=np.float32, buffer=existing_buff1_shm.buf)
    buff2 = np.ndarray(((window_length, nb_sensors)), dtype=np.float32, buffer=existing_buff2_shm.buf)
    buff3 = np.ndarray(((window_length, nb_sensors)), dtype=np.float32, buffer=existing_buff3_shm.buf)

    y_pred = []
    y_pred_maj = []
    inference_time = []
    maj_vote_time = []
    total_process_time = []


    time_before_inference= 0
    previous_majority_vote_time = 0
    inference_index = 0

    # Create interpreter
    interpreter = infer.make_interpreter(model_path)
    #TODO check if correct
    width, height = common.input_size(interpreter)
    interpreter.allocate_tensors()
    infer.make_inference(interpreter, np.random.rand(nb_sensors).reshape(width,height,1))

    nb_votes = int(np.floor(vote_length/window_length))
    votes_arr = np.zeros(nb_votes, dtype=np.uint8)
    curr_vote = 0
    while done[0] == False:
        inference_ready = mutex[0]
        if inference_ready > 0:
            processing_start = time.perf_counter()
            if debug:
                print("Real inference : %s"%(inference_index))
                inference_index += 1
                time_inf = time.time()-time_before_inference
                print("Time before inference: %.1fms"%(time_inf*1000))

            lock.acquire()
            mutex[0] -= 1
            lock.release()
            read_buffer = mutex[2]
            if read_buffer == 0:
                data = dp.process_buffer(buff1, fs=1000, Q=30, notch_freq=60, filtering_utility=filtering_utility)
            elif read_buffer == 1:
                data = dp.process_buffer(buff2, fs=1000, Q=30, notch_freq=60, filtering_utility=filtering_utility)
            elif read_buffer == 2:
                data = dp.process_buffer(buff3, fs=1000, Q=30, notch_freq=60, filtering_utility=filtering_utility)

            scaled_data = dp.compress_data(data.reshape(1,nb_sensors), compression_method, residual_bits).reshape(width,height,1)
            inference_start = time.perf_counter()
            vote = infer.make_inference(interpreter, scaled_data, False)
            inf_time = time.perf_counter()-inference_start

            votes_arr[curr_vote] = vote

            if (curr_vote+1 == nb_votes):
                current_majority_vote_time = time.time()
                time_before_majority_vote = current_majority_vote_time-previous_majority_vote_time
                previous_majority_vote_time = current_majority_vote_time
                majority_vote = np.argmax(np.bincount(votes_arr))
                y_pred_maj.append(majority_vote)
                maj_vote_time.append(time_before_majority_vote)
                if debug:
                    print("Time before majority vote: %.1fms"%(time_before_majority_vote*1000))
                    print("Majority vote: %s"%(majority_vote))

            curr_vote = (curr_vote+1)%nb_votes
            time_before_inference = time.time()

            # Increment read buffer index
            lock.acquire()
            mutex[2] = (read_buffer+1)%3
            lock.release()

            processing_end = time.perf_counter()

            # Append results to list
            total_process_time.append(processing_end-processing_start)
            inference_time.append(inf_time)
            y_pred.append(vote)

    queue.put(y_pred)
    queue.put(y_pred_maj)
    queue.put(inference_time)
    queue.put(maj_vote_time)
    queue.put(total_process_time)

def model_evaluation(dataset, model_name, subject, session, compression_method, residual_bits, fine_tuned=False, on_device=False, debug=False):
    model_name = "%s_%s_%s_%s_%s_%sbits"%(dataset.name, model_name, subject, session, compression_method, residual_bits)
    nb_sensors = dataset.sensors_dim[0]*dataset.sensors_dim[1]
    filtering_utility= dataset.utility_filtered
    sampling_rate = (1/dataset.sampling_rate)

    test_session = "2" if session == "1" else "1"

    # Create data
    dataset_path = "/home/mendel/dataset/%s/%s_%s_raw.npz"%(dataset.name, subject, test_session)

    model_accuracy = []
    model_accuracy_maj = []
    confusion_list = []
    confusion_list_maj = []

    model_inference_time = []
    model_maj_vote_time = []
    model_total_process_time = []

    for i in range(5):
        queue = mp.Queue()
        # Create a range of 2 values to select fine tuning data based on i [0,1]; [2,3] ...
        fine_tuning_range = range(i*2, i*2+2)
        if fine_tuned:
            if on_device:
                running_model_name = model_name + "_ondevice_tuned_%s_%s"%(fine_tuning_range[0], fine_tuning_range[-1])
            else:
                running_model_name = model_name + "_tuned_%s_%s"%(fine_tuning_range[0], fine_tuning_range[-1])
        else:
            running_model_name = model_name
        model_path = "/home/mendel/model/%s_edgetpu.tflite"%(running_model_name)
        #get all the rest of the range for testing from  0 to 9
        testing_range = list(set(range(10)) - set(fine_tuning_range))
        X, y = create_sampled_data(dataset_path, testing_range)

        # Get majority vote for y in the time window
        nb_votes = int(np.floor(_VOTE_LENGTH/_TIME_LENGTH))

        y_true = np.array([np.argmax(np.bincount(y[i:i+_TIME_LENGTH])) for i in range(0, len(y), _TIME_LENGTH)])
        y_true_maj = np.array([np.argmax(np.bincount(y_true[i:i+nb_votes])) for i in range(0, len(y_true), nb_votes)])

        # Create shared memory
        done_shm, mutex_shm, buff_1_shm, buff_2_shm, buff_3_shm  = create_shared_memory(_TIME_LENGTH, nb_sensors)
        data_sampling = mp.Process(name="adc_sampling", target=sample_data, args=(X, nb_sensors, sampling_rate,_TIME_LENGTH, done_shm.name, mutex_shm.name, buff_1_shm.name, buff_2_shm.name, buff_3_shm.name, debug))
        inference = mp.Process(name="main_process", target=inference_process, args=(queue, model_path, nb_sensors, filtering_utility, compression_method, residual_bits, _VOTE_LENGTH, _TIME_LENGTH, done_shm.name, mutex_shm.name, buff_1_shm.name, buff_2_shm.name, buff_3_shm.name, debug))
        inference.start()
        time.sleep(10)
        data_sampling.start()

        y_pred = queue.get()
        y_pred_maj = queue.get()
        inference_time = queue.get()
        maj_vote_time = queue.get()
        total_process_time = queue.get()
        
        data_sampling.join()
        inference.join()

        print("done")
        inference.close()
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

        if debug:
            print("Number of output labels:", len(y_pred))
            print("Number of majority vote labels:", len(y_pred_maj))

        y_pred = np.array(y_pred, dtype=np.uint8)
        y_pred_maj = np.array(y_pred_maj, dtype=np.uint8)

        # Evaluate model performance
        accuracy = metrics.accuracy_score(y_true, y_pred)
        accuracy_maj = metrics.accuracy_score(y_true_maj, y_pred_maj)

        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        confusion_matrix_maj = metrics.confusion_matrix(y_true_maj, y_pred_maj)

        print("Accuracy: %.2f%%"%(accuracy*100))
        print("Accuracy majority vote: %.2f%%"%(accuracy_maj*100))
        model_accuracy.append(accuracy)
        model_accuracy_maj.append(accuracy_maj)
        confusion_list.append(confusion_matrix)
        confusion_list_maj.append(confusion_matrix_maj)
        model_inference_time.append(inference_time)
        model_maj_vote_time.append(maj_vote_time)
        model_total_process_time.append(total_process_time)


    if fine_tuned:
        if on_device:
            filename = "/home/mendel/results/%s_realtime_evaluation_ondevice.npz"%(model_name)
        else:
            filename = "/home/mendel/results/%s_realtime_evaluation_finetuned.npz"%(model_name)
    else:
        filename = "/home/mendel/results/%s_realtime_evaluation.npz"%(model_name)
    np.savez(filename, accuracy=np.array(model_accuracy), accuracy_majority_vote=np.array(model_accuracy_maj), 
             confusion_matrix=np.array(confusion_list), confusion_matrix_maj=np.array(confusion_list_maj), 
             inference_time=np.array(model_inference_time), maj_vote_time=np.array(model_maj_vote_time), process_time=np.array(model_total_process_time))


if __name__ == '__main__':
    dataset = dtdef.CapgmyoDataset()
    subject = "01"
    model_name = "cnn"
    residual_bits = 8
    model_evaluation(dataset, model_name, subject, "1", "minmax", residual_bits, fine_tuned=False, on_device=False, debug=True)
    model_evaluation(dataset, model_name, subject, "1", "msb",    residual_bits, fine_tuned=False, on_device=False, debug=False)
    model_evaluation(dataset, model_name, subject, "1", "smart",  residual_bits, fine_tuned=False, on_device=False, debug=False)
    model_evaluation(dataset, model_name, subject, "1", "root",   residual_bits, fine_tuned=False, on_device=False, debug=False)
    model_evaluation(dataset, model_name, subject, "2", "minmax", residual_bits, fine_tuned=False, on_device=False, debug=False)
    model_evaluation(dataset, model_name, subject, "2", "msb",    residual_bits, fine_tuned=False, on_device=False, debug=False)
    model_evaluation(dataset, model_name, subject, "2", "smart",  residual_bits, fine_tuned=False, on_device=False, debug=False)
    model_evaluation(dataset, model_name, subject, "2", "root",   residual_bits, fine_tuned=False, on_device=False, debug=False)

    # model_evaluation(dataset, model_name, subject, "1", "minmax", residual_bits, fine_tuned=True, on_device=False, debug=False)
    # model_evaluation(dataset, model_name, subject, "1", "msb",    residual_bits, fine_tuned=True, on_device=False, debug=False)
    # model_evaluation(dataset, model_name, subject, "1", "smart",  residual_bits, fine_tuned=True, on_device=False, debug=False)
    # model_evaluation(dataset, model_name, subject, "1", "root",   residual_bits, fine_tuned=True, on_device=False, debug=False)
    # model_evaluation(dataset, model_name, subject, "2", "minmax", residual_bits, fine_tuned=True, on_device=False, debug=False)
    # model_evaluation(dataset, model_name, subject, "2", "msb",    residual_bits, fine_tuned=True, on_device=False, debug=False)
    # model_evaluation(dataset, model_name, subject, "2", "smart",  residual_bits, fine_tuned=True, on_device=False, debug=False)
    # model_evaluation(dataset, model_name, subject, "2", "root",   residual_bits, fine_tuned=True, on_device=False, debug=False)

    # model_evaluation(dataset, model_name, subject, "1", "minmax", residual_bits, fine_tuned=True, on_device=True, debug=False)
    # model_evaluation(dataset, model_name, subject, "1", "msb",    residual_bits, fine_tuned=True, on_device=True, debug=False)
    # model_evaluation(dataset, model_name, subject, "1", "smart",  residual_bits, fine_tuned=True, on_device=True, debug=False)
    # model_evaluation(dataset, model_name, subject, "1", "root",   residual_bits, fine_tuned=True, on_device=True, debug=False)
    # model_evaluation(dataset, model_name, subject, "2", "minmax", residual_bits, fine_tuned=True, on_device=True, debug=False)
    # model_evaluation(dataset, model_name, subject, "2", "msb",    residual_bits, fine_tuned=True, on_device=True, debug=False)
    # model_evaluation(dataset, model_name, subject, "2", "smart",  residual_bits, fine_tuned=True, on_device=True, debug=False)
    # model_evaluation(dataset, model_name, subject, "2", "root",   residual_bits, fine_tuned=True, on_device=True, debug=False)

