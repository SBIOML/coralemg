import os
import re
import fnmatch
import numpy as np
import scipy.io as sio
from collections import Counter
import data_processing as dp


def load_emager(path, subject, session, differential=False):
    """
    Load EMG data from EMaGer v1 dataset.

    Params:
        - path : path to EMaGer root
        - subject : subject id
        - session : session number

    Returns the loaded data with shape (nb_gesture, nb_repetition, time_length, nb_channels)
    """

    # Parameters
    # subject = "01"
    # session = "1"
    nb_gesture = 6
    nb_repetition = 10
    nb_pts = 5000
    start_path = "%s/subject_%s/session_%s/" % (path, subject, session)  # ordi UL
    data_array = np.zeros((nb_gesture, nb_repetition, 64, nb_pts), dtype=int)

    first_file = os.listdir(start_path)[0]
    arm_used = "right" if "right" in first_file else "left"
    for gest in range(nb_gesture):
        for rep in range(nb_repetition):
            path = (
                start_path
                + "0"
                + subject
                + "-00"
                + session
                + "-00"
                + str(gest)
                + "-00"
                + str(rep)
                + "-"
                + arm_used
                + ".csv"
            )
            one_file = np.transpose(np.loadtxt(path, delimiter=","))
            data_array[gest, rep, :, :] = one_file[:, -nb_pts:]
    if differential:
        data_array = np.reshape(data_array, (nb_gesture, nb_repetition, 16, 4, nb_pts))
        final_array = data_array[:, :, :, 0:3, :] - data_array[:, :, :, 1:4, :]
        final_array = np.reshape(final_array, (nb_gesture, nb_repetition, 48, nb_pts))
    else:
        final_array = data_array

    return np.swapaxes(final_array, 2, 3)

def load_capgmyo(path, subject, session):
    """
    Load EMG data from CapgMyo dataset.
    The format of the folder has to be 

    capgmyo
├── subject01_session1
│   ├── 001-001.mat
│   ├── 001-002.mat
...

    Params:
        - path : path to Capgmyo root folder
        - subject : subject id
        - session : session number

    Returns the loaded data with shape (nb_gesture, nb_repetition, time_length, nb_channels)
    """

    # Parameters
    # subject = "01"
    # session = "1"
    dirpath = "%s/subject%s_session%s/" %(path, subject, session)
    files = fnmatch.filter(os.listdir(dirpath), '*.mat')
    files = np.sort(files)
    experiment_list = [] # TODO : See if better method exists
    for file in files:
        experiment = sio.loadmat(dirpath+file)

        experiment_list.append(experiment)

    experiment_array = _capgmyo_format_array(experiment_list)
    
    return experiment_array

def _capgmyo_format_array(experiment_list):
    exp_0 = experiment_list[0]
    data_length, nb_channels = np.shape(exp_0['data'])
    label_list = []
    for i, experiment in enumerate(experiment_list):
        label_list.append(experiment['gesture'][0][0])
    count = Counter(label_list)

    nb_labels = len(count.values())
    nb_exp = max(count.values())

    data_array = np.full((nb_labels, nb_exp, data_length, nb_channels), None)
    curr_exp = 0
    previous_label = -1
    for i, experiment in enumerate(experiment_list):
        curr_data = experiment['data']
        label = experiment['gesture']
        if label != previous_label:
            curr_exp = 0
        else :
            curr_exp += 1
        data_array[label-1,curr_exp,:,:] = curr_data[:,:]
        previous_label = label
    return data_array

def load_hyser(path, subject, session):
    """
    Load EMG data from Hyser dataset.

    Params:
        - path : path to Hyser root
        - subject : subject id
        - session : session number

    Returns the loaded data with shape (nb_gesture, nb_repetition, time_length, nb_channels)
    """

    # Parameters
    # subject = "01"
    # session = "1"

    # Compiled regex used for gain and baseline
    RE_GAIN = re.compile(".+?(?=\()")
    RE_BASELINE = re.compile("(?<=\().*(?=\))")

    task_type = "maintenance"
    sig_type = "preprocess"

    dirpath = "%s/subject%s_session%s/" %(path, subject, session)
    nb_files = len(fnmatch.filter(os.listdir(dirpath), '%s_%s_*.dat'%(task_type, sig_type)))

    data = [] # TODO : See if better method exists

    file_name = "%slabel_%s.txt"%(dirpath, task_type)
    with open(file_name) as fid :
        labels = np.loadtxt(fid, delimiter =',', dtype = int)

    for i in range(1,nb_files+1):
        file_name = "%s%s_%s_sample%s"%(dirpath, task_type, sig_type, i)
        # Import data
        with open(file_name+".dat", 'rb') as fid:
            data_array = np.fromfile(fid, dtype=np.int16).reshape(-1,256).astype('float')
        
        # Import gain and baseline and applies it to data
        with open(file_name+".hea",'r') as fid:
            head_info = np.char.split((fid.readlines()[1:]))
            for j,line in enumerate(head_info):
                str_tmp = line[2]
                gain = float(RE_GAIN.match(str_tmp).group())
                baseline = float(RE_BASELINE.search(str_tmp).group())
                data_array[:,j] = (data_array[:,j]-baseline)/gain
        
        # Reshape the data
        data_array = data_array.reshape(-1,4,8,8) 
        data_array = np.flip(data_array,(2,3))
        data_array[:,[1,2],:,:] = data_array[:,[2,1],:,:]


        data_array = np.concatenate((np.concatenate((data_array[:,0,:,:],data_array[:,1,:,:]),axis = 2),np.concatenate((data_array[:,2,:,:],data_array[:,3,:,:]),axis = 2)),axis=1).reshape(-1,256)
        
        data.append(data_array.reshape(-1,256)) # TODO : See if better method exists
    
    nb_experiment, time_length, nb_channels = np.shape(data)
    nb_gesture = len(set(labels))
    nb_repetition = int(nb_experiment/nb_gesture)

    # First 512 data points are noise
    time_length = time_length-512
    experiment_array = np.zeros((nb_gesture, nb_repetition, time_length, nb_channels))

    for i, experiment in enumerate(data):
        data[i] = experiment[512:,:]
        curr_label = int(labels[i]-1)
        experiment_array[curr_label, i%nb_repetition,:,:] = data[i]

    return experiment_array

def load_dataset(path, subject, session):
    """
    Load EMG data from given dataset.

    Params:
        - path : path to EMG dataset
        - subject : subject id
        - session : session number

    Returns the loaded data with shape (nb_gesture, nb_repetition, time_length, nb_channels)
    """
    if "emager" in path:
        experiment_array = load_emager(path, subject, session, differential=False)
    elif "capgmyo" in path:
        experiment_array = dp.convert_capgmyo_16bit(load_capgmyo(path, subject, session))
    elif "hyser" in path:
        experiment_array = dp.convert_hyser_16bit(load_hyser(path, subject, session))
    else:
        raise Exception("Supported dataset is not in path")
    
    return experiment_array


def save_training_data(
    dataset_path, subject, session, compressed_method="minmax", nb_bits=8, window_length=25, dimension=(4,16), save_folder_path=""
):
    """
    Save the training data for the tensorflow model

    @param dataset_path the path to the dataset
    @param subject the subject to use, must be 000, 001, ...
    @param session the session to use, must be 001, 002
    @param compressed_method the compression method used
    @param nb_bits the number of bits to compress to
    @param window_length the length of the time window to use
    @param dimension the number of electrode in each axis (x,y)
    @param save_folder_path the path of the folder to save the data in

    """
    main_folder_path = save_folder_path
    if not os.path.exists(main_folder_path):
        os.makedirs(main_folder_path)

    data_array = load_dataset(dataset_path, subject, session)

    filtering_utility = True if "emager" in dataset_path else False
    averages_data = dp.preprocess_data(data_array, window_length=window_length, filtering_utility=filtering_utility)
    X, y = dp.extract_with_labels(averages_data)

    X_compressed = dp.compress_data(X, method=compressed_method, residual_bits=nb_bits)
    filename = "%s/%s_%s_%s_%sbits.npz" % (
        main_folder_path,
        subject,
        session,
        compressed_method,
        nb_bits
    )
    X_rolled = dp.roll_data(X_compressed, 2, v_dim=dimension[0], h_dim=dimension[1])

    # Copy the labels to be the same size as the data
    nb_rolled = int(np.floor(X_rolled.shape[0] / y.shape[0]))
    y_rolled = np.tile(y, nb_rolled)
    y_rolled = np.array(y_rolled, dtype=np.uint8)

    np.savez(filename, data=X_rolled, label=y_rolled)


def save_raw_data(dataset_path, subject, session, save_folder_path="dataset/raw/"):
    """
    Save the training data for the tensorflow model

    @param dataset_path the path to the dataset
    @param subject the subject to use, must be 000, 001, ...
    @param session the session to use, must be 001, 002
    @param save_folder_path the path of the folder to save the data in

    """
    main_folder_path = save_folder_path
    if not os.path.exists(main_folder_path):
        os.makedirs(main_folder_path)

    filename = "%s/%s_%s_raw.npz" % (main_folder_path, subject, session)
    data_array = load_dataset(dataset_path, subject, session)
    np.savez(filename, data=data_array)


if __name__ == "__main__":
    # dataset_path = "dataset/emager"
    # subjects = ["00","01","02","03","04","05","06","07","08","09","10", "11"]
    # sessions = ["1", "2"]
    # bits = [4,5,6,7,8]
    # compressed_methods = ["minmax", "msb", "smart", "root", "baseline"]
    # #compressed_methods = ["baseline"]

    # for subject in subjects:
    #     for session in sessions:
    #         for compressed_method in compressed_methods:
    #             for bit in bits:
    #                 save_training_data(
    #                     dataset_path,
    #                     subject,
    #                     session,
    #                     compressed_method=compressed_method,
    #                     nb_bits=bit,
    #                     save_folder_path="dataset/train/emager/%s" % (compressed_method),
    #                 )
    #         save_raw_data(dataset_path, subject, session, "dataset/raw/emager/")

    dataset_path = "dataset/capgmyo"
    subjects = ["01","02","03","04","05","06","07","08","09","10"]
    sessions = ["1", "2"]
    bits = [4,5,6,7,8]
    compressed_methods = ["minmax", "msb", "smart", "root", "baseline"]
    #compressed_methods = ["baseline"]

    for subject in subjects:
        for session in sessions:
            for compressed_method in compressed_methods:
                for bit in bits:
                    save_training_data(
                        dataset_path,
                        subject,
                        session,
                        compressed_method=compressed_method,
                        nb_bits=bit,
                        dimension=(8,16),
                        save_folder_path="dataset/train/capgmyo/%s" % (compressed_method),
                    )
            save_raw_data(dataset_path, subject, session, "dataset/raw/capgmyo/")