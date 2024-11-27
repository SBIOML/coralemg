from abc import ABC, abstractmethod
import os
import re
import fnmatch
import numpy as np
import scipy.io as sio
from collections import Counter

class EmgDataset(ABC):
    def __init__(self):
        self.name = None
        self.sampling_rate = None
        self.sensors_dim = None
        self.nb_experiment = None
        self.nb_class = None
        self.utility_filtered = None

    @abstractmethod
    def convert_to_16bits(self, experiment_array):
        pass

    @abstractmethod
    def load_dataset(self, path, subject, session):
        pass

class EmagerDataset(EmgDataset):
    def __init__(self, differential=False):
        self.name = "emager"
        self.sampling_rate = 1000
        self.sensors_dim = (4,16)
        self.nb_experiment = 10
        self.nb_class = 6
        self.utility_filtered = False
        self.differential = differential

    def load_emager(self, path, subject, session, differential=False):
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
    
    def convert_to_16bits(self, experiment_array):
        return experiment_array.astype(np.int16)

    def load_dataset(self, path, subject, session):
        return self.convert_to_16bits(self.load_emager(path, subject, session, self.differential))
    
class CapgmyoDataset(EmgDataset):
    def __init__(self):
        self.name = "capgmyo"
        self.sampling_rate = 1000
        self.sensors_dim = (8,16)
        self.nb_experiment = 10
        self.nb_class = 8
        self.utility_filtered = True
    
    def load_capgmyo(self, path, subject, session):
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

        experiment_array = self._capgmyo_format_array(experiment_list)
        
        return experiment_array

    def _capgmyo_format_array(self, experiment_list):
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
    
    def convert_to_16bits(self, experiment_array):
        return np.floor(experiment_array*32767).astype(np.int16)
    
    def load_dataset(self, path, subject, session):
        return self.convert_to_16bits(self.load_capgmyo(path, subject, session))
    
class HyserDataset(EmgDataset):
    def __init__(self):
        self.name = "hyser"
        self.sampling_rate = 2000 #(should be 2048 but 2000 is used for nb of samples purpose)
        self.sensors_dim = (16,16)
        self.nb_experiment = 2
        self.nb_class = 34
        self.utility_filtered = True

    def load_hyser(self, path, subject, session):
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

        dim = np.prod(self.sensors_dim)

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
                data_array = np.fromfile(fid, dtype=np.int16).reshape(-1,dim).astype('float')
            
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


            data_array = np.concatenate((np.concatenate((data_array[:,0,:,:],data_array[:,1,:,:]),axis = 2),
                                         np.concatenate((data_array[:,2,:,:],data_array[:,3,:,:]),axis = 2)),
                                         axis=1).reshape(-1,256)
            
            data.append(data_array.reshape(-1,dim)) # TODO : See if better method exists
        
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
        
    def convert_to_16bits(self, experiment_array):
        return np.floor(experiment_array*32767).astype(np.int16)

    def load_dataset(self, path, subject, session):
        return self.convert_to_16bits(self.load_hyser(path, subject, session))
