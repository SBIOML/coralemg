import numpy as np
from scipy import signal
import quantization as compress


def filter_utility(data, fs=1000, Q=30, notch_freq=60):
    b_notch, a_notch = signal.iirnotch(notch_freq, Q, fs)
    return signal.filtfilt(b_notch, a_notch, data, axis=0)

def extract_with_labels(data_array):
    """
    Given a data array, it will extract the data and labels from the array.

    @param data_array the data array to extract from

    @return a tuple of (data, labels) arrays
    """
    labels, nb_exp, nb_sample, nb_channels = np.shape(data_array)
    X = np.zeros((labels * nb_exp * nb_sample, nb_channels))
    y = np.zeros((labels * nb_exp * nb_sample))
    for label in range(labels):
        for experiment in range(nb_exp):
            X[
                nb_sample
                * (label * nb_exp + experiment) : nb_sample
                * (label * nb_exp + experiment + 1),
                :,
            ] = data_array[label, experiment, :, :]
            y[
                nb_sample
                * (label * nb_exp + experiment) : nb_sample
                * (label * nb_exp + experiment + 1)
            ] = label
    return X, y

def process_buffer(buffer, fs=1000, Q=30, notch_freq=60):
    processed_data = filter_utility(buffer, fs=fs, Q=Q, notch_freq=notch_freq)
    return np.mean(np.absolute(processed_data - np.mean(processed_data,axis=0)),axis=0)

def compress_data(data, method="minmax", residual_bits=8):
    """
    Given a data array, it will compress the data by the specified method.

    @param data the data array to be compressed. The data is assumed to be int16
    @param method the method to use for compression, can be "minmax", "msb", or "smart"
    @param residual_bits the number of bits to compress to

    @return the compressed data array
    """
    if method == "minmax":
        return compress.normalize_min_max_c(data, residual_bits).astype(np.uint8)
    elif method == "msb":
        return compress.naive_bitshift_c(data, residual_bits).astype(np.uint8)
    elif method == "smart":
        msb_shift = (8-residual_bits)+3
        return compress.smart_bitshift_c(data, residual_bits, msb_shift).astype(np.uint8)
    elif method == "log":
        return compress.log_c(data, 20000, residual_bits).astype(np.uint8)
    elif method == "root":
        return compress.nroot_c(data, 3.0, 20000, residual_bits).astype(np.uint8)
    else:
        raise ValueError("Invalid compression method")

def convert_capgmyo_16bit(experiment_array):
    return np.floor(experiment_array*32767).astype(np.int16)

def convert_hyser_16bit(experiment_array):
    pass
