import numpy as np
from scipy import signal
import save_dataset as sd
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


def preprocess_data(data_array, window_length=25, fs=1000, Q=30, notch_freq=60, filtering_utility=False):
    """
    Given a data array, it will preprocess the data by applying the desired operations.

    @param data: the data array to be processed, the data array has the format (nb_gesture, nb_repetition, time_length, num_channels)
    @param window_length the length of the time window to use
    @param fs the sampling frequency of the data
    @param Q the quality factor of the notch filter
    @param notch_freq the frequency of the notch filter

    @return the processed data array
    """
    labels, nb_exp, total_time_length, nb_channels = np.shape(data_array)

    nb_window = int(np.floor(total_time_length / window_length))
    output_data = np.zeros((labels, nb_exp, nb_window, nb_channels))

    for label in range(labels):
        for experiment in range(nb_exp):
            for curr_window in range(nb_window):
                start = curr_window * window_length
                end = (curr_window + 1) * window_length
                processed_data = data_array[label, experiment, start:end, :]
                if filtering_utility:
                    processed_data = filter_utility(
                        processed_data, fs=fs, Q=Q, notch_freq=notch_freq
                    )
                processed_data = np.mean(
                    np.absolute(processed_data - np.mean(processed_data, axis=0)),
                    axis=0,
                )
                output_data[label, experiment, curr_window, :] = processed_data
    return output_data


def roll_data(data_array, rolled_range, v_dim=4, h_dim=16):
    """
    Given a data array, it will roll the data by the specified amount.

    @param data the data array to be rolled
    @param rolled_range the amount to roll the data by

    @return the rolled data array
    """
    nb_sample, nb_channels = np.shape(data_array)
    roll_index = range(-rolled_range, rolled_range + 1)
    nb_out = len(roll_index) * nb_sample

    output_data = np.zeros((nb_out, nb_channels))
    for i, roll in enumerate(roll_index):
        tmp_data = _roll_array(data_array, roll, v_dim, h_dim)
        output_data[i * nb_sample : (i + 1) * nb_sample, :] = tmp_data
    return output_data


def _roll_array(
    data,
    roll,
    v_dim=4,
    h_dim=16,
):
    tmp_data = np.array(data)
    tmp_data = np.reshape(tmp_data, (-1, v_dim, h_dim))
    tmp_data = np.roll(tmp_data, roll, axis=2)
    tmp_data = np.reshape(tmp_data, (-1, v_dim * h_dim))
    return tmp_data


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
    elif method == "baseline":
        return data
    else:
        raise ValueError("Invalid compression method")

def convert_capgmyo_16bit(experiment_array):
    return np.floor(experiment_array*32767).astype(np.int16)

def convert_hyser_16bit(experiment_array):
    pass

if __name__ == "__main__":
    # generate sinus of 60 hz
    dataset_path = "dataset/emager/"
    data_array = sd.load_emager(dataset_path, "000", "002", differential=False)
    print(data_array.shape)
    # preprocess the data
    averages = preprocess_data(data_array)
    print(np.shape(averages))

    # Visualize the data

    # roll the data
    rolled = roll_data(averages, 2)
    print(np.shape(rolled))
    X, y = extract_with_labels(data_array)

    y = np.array(y, dtype=np.uint8)
    print(np.shape(X))
    _TIME_LENGTH = 25
    _VOTE_LENGTH = 150
    nb_votes = int(np.floor(_VOTE_LENGTH / _TIME_LENGTH))

    expected = np.array(
        [
            np.argmax(np.bincount(y[i : i + _TIME_LENGTH]))
            for i in range(0, len(y), _TIME_LENGTH)
        ]
    )
    maj_expected = np.array(
        [
            np.argmax(np.bincount(expected[i : i + nb_votes]))
            for i in range(0, len(expected), nb_votes)
        ]
    )

    print(np.shape(expected))
    print(np.shape(maj_expected))

    """
    example usage (sd.save_training_data)
    data_array = load_emager(dataset_path, subject, session, differential=False)
    averages_data = dp.preprocess_data(data_array)
    compressed_data = dp.compress_data(averages_data, method=compressed_method)
    rolled_data = dp.roll_data(compressed_data, 2)
    X, y = dp.extract_with_labels(rolled_data)
    """
