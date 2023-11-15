import os
import numpy as np
import re
import fnmatch
from sklearn import preprocessing

def getData_EMG(user_id, session_nb, arm_used = "right", differential=False):
    # Parameters
    #user_id = "001"
    #session_nb = "000"
    nb_gesture = 6
    nb_repetition = 10
    nb_pts = 5000
    # start_path = 'C:/Users/felix/OneDrive/Documents/SCHOOL/ETE2022/Projet/Dataset/user_001/session_000/'  # ordi perso
    start_path = '/home/etienne/Documents/Universite/maitrise/recherche/GSPEMG_OLD/ULEMG/%s/session_%s/'%(user_id, session_nb)  # ordi UL
    data_array = np.zeros((nb_gesture, nb_repetition, 64, nb_pts), dtype=int)
    for gest in range(nb_gesture):
        for rep in range(nb_repetition):
            path = start_path + user_id + "-" + session_nb + "-00" + str(gest) + "-00" + str(rep) + "-" + arm_used + ".csv"
            one_file = np.transpose(np.loadtxt(path, delimiter=','))
            data_array[gest, rep, :, :] = one_file[:, -nb_pts:]
    if differential:
        data_array = np.reshape(data_array, (nb_gesture, nb_repetition,16,4,nb_pts))
        final_array = data_array[:,:,:,0:3,:] - data_array[:,:,:,1:4,:]
        final_array = np.reshape(final_array,(nb_gesture, nb_repetition,48,nb_pts))
    else:
        final_array = data_array
        print("Out of the function:", final_array)

    return np.swapaxes(final_array, 2, 3)

# Compiled regex used for gain and baseline
RE_GAIN = re.compile(".+?(?=\()")
RE_BASELINE = re.compile("(?<=\().*(?=\))")

# Compiled regex to know the subject and session
re_subject = re.compile(r'(?<=subject)[^\s]+(?=_)')
re_session = re.compile(r'(?<=session)[^\s]*')


def save_data_array_for_edge(task_type = "maintenance", sig_type = "preprocess"):
    subject = "000"
    session = "001"

    str_channels = "64"

    filename = "tflite_data_test.npz"

    experiment_array = getData_EMG(subject, session, arm_used = "left", differential=False)
    labels, nb_exp, total_time_length, nb_channels = np.shape(experiment_array)

    print(np.shape(experiment_array))

    output_data = np.zeros((3000,64))
    output_label = np.zeros(3000)

    output_data[0:1500,:] = experiment_array[0,0,0:1500,:]
    output_label[0:1500] = np.ones(1500)*0

    output_data[1500:3000,:] = experiment_array[1,0,0:1500,:]
    output_label[1500:3000] = np.ones(1500)*1

    np.savez(filename,data=output_data, label=output_label)


def save_average(subject, train_sessions, test_sessions, time_length = 25, rolled = False, arm_used = "left", folder_name = 'ftensorflow'):

    subject = "000"
    sessions = ["001","002", "003","004","006","009","010"]


    str_channels = "64"

    main_folder_path = "dataset/average/%s" %(folder_name)
    features_folder_path = "%s/chan_%s/window_%s" %(main_folder_path, str_channels, time_length)

    if not os.path.exists(features_folder_path):
        os.makedirs(features_folder_path)

    with open('%s/parameters.txt' %(main_folder_path), 'w') as fp:
        pass

    filename = "%s/all_subject.npz"%(features_folder_path)
    train_data = np.zeros((1,4,16), dtype=np.uint8)
    train_label = np.zeros(1, dtype=np.uint8)
    test_data = np.zeros((1,4,16), dtype=np.uint8)
    test_label = np.zeros(1, dtype=np.uint8)

    first_value = True
    for session in train_sessions:
        experiment_array = getData_EMG(subject, session, arm_used = arm_used, differential=False)
        labels, nb_exp, total_time_length, nb_channels = np.shape(experiment_array)
        for label in range(labels):
            curr_label = np.zeros(1, dtype=np.uint8)
            curr_label[0] = label
            for experiment in range(nb_exp):
                nb_samples = int(np.floor(total_time_length/time_length))
                for sample in range(nb_samples):
                    if rolled:
                        for roll_idx in range(-2,3):
                            roll_str = "m%s"%(np.abs(roll_idx)) if roll_idx < 0 else "p%s"%(np.abs(roll_idx))
                            data_sample = experiment_array[label,experiment, time_length*sample:time_length*sample+time_length,:]
                            tmp_data = np.reshape(data_sample,(-1,4,16))
                            tmp_data = np.roll(tmp_data,roll_idx, axis=2)
                            data_sample = np.reshape(tmp_data,(-1,64))
                            data_mean = np.mean(np.absolute(data_sample - np.mean(data_sample,axis=0)),axis=0)
                            data_mean = np.uint8(preprocessing.minmax_scale(data_mean)*255).reshape(1,4,16)
                            if first_value:
                                train_data[0,:,:] = data_mean
                                train_label[0] = curr_label[0]
                                first_value = False
                            else:
                                train_data = np.concatenate((train_data, data_mean), dtype=np.uint8)
                                train_label = np.concatenate((train_label, curr_label), dtype=np.uint8)
                    else:
                        data_sample = experiment_array[label,experiment, time_length*sample:time_length*sample+time_length,:]
                        data_mean = np.mean(np.absolute(data_sample - np.mean(data_sample,axis=0)),axis=0)
                        data_mean = np.uint8(preprocessing.minmax_scale(data_mean)*255).reshape(1,4,16)
                        if first_value:
                            train_data[0,:,:] = data_mean
                            train_label[0] = curr_label[0]
                            first_value = False
                        else:
                            train_data = np.concatenate((train_data, data_mean), dtype=np.uint8)
                            train_label = np.concatenate((train_label, curr_label), dtype=np.uint8)

    first_value = True
    for session in test_sessions:
        experiment_array = getData_EMG(subject, session, arm_used = arm_used, differential=False)
        labels, nb_exp, total_time_length, nb_channels = np.shape(experiment_array)
        for label in range(labels):
            curr_label = np.zeros(1, dtype=np.uint8)
            curr_label[0] = label
            for experiment in range(nb_exp):
                nb_samples = int(np.floor(total_time_length/time_length))
                for sample in range(nb_samples):
                    data_sample = experiment_array[label,experiment, time_length*sample:time_length*sample+time_length,:]
                    data_mean = np.mean(np.absolute(data_sample - np.mean(data_sample,axis=0)),axis=0)
                    data_mean = np.uint8(preprocessing.minmax_scale(data_mean)*255).reshape(1,4,16)
                    if first_value:
                        test_data[0,:,:] = data_mean
                        test_label[0] = curr_label[0]
                        first_value = False
                    else:
                        test_data = np.concatenate((test_data, data_mean), dtype=np.uint8)
                        test_label = np.concatenate((test_label, curr_label), dtype=np.uint8)


    np.savez(filename,train_data=train_data, train_label=train_label, test_data=test_data, test_label=test_label)

if __name__ == '__main__':
    subject = "000"
    train_sessions = ["003","004","006","009","010"]
    test_sessions = ["001","002"]
    save_data_array_for_edge()
    #save_average(subject, train_sessions, test_sessions, rolled=True)
