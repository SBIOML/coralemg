import numpy as np
from scipy import signal
from scipy.io import loadmat
from tqdm import tqdm
import json
import os

from emager_py.data import dataset


def throughput_bits(bits, fs=1000):
    """Return the EMG throughput in kbps

    Args:
        bits (int): number of bits per sample
        fs (int, optional): EMG sampling rate. Defaults to 1000.

    Returns:
        float: throughput in kbps
    """
    return 64 * bits * fs / 1000


def iter_emager(dataset_root: str, preprocess=True, analog=False):
    if not dataset_root.endswith("/"):
        dataset_root += "/"

    for subject in dataset.get_subjects(dataset_root):
        for session in dataset.get_sessions():
            data = dataset.load_emager_data(
                dataset_root,
                subject,
                session,
            )
            if preprocess:
                data = prefilter(data)
            if analog:
                data = data * 0.195e-3  # convert to mV
            for gesture in range(data.shape[0]):
                for trial in range(data.shape[1]):
                    d = data[gesture, trial]
                    yield d


def load_emager(dataset_root: str, preprocess=True, analog=False):
    x = []
    for i in iter_emager(dataset_root, preprocess=preprocess, analog=analog):
        x.append(i)
    data = np.array(x).reshape(-1, 64)
    return data.astype(np.float32)


_notch = signal.tf2sos(*signal.iirnotch(60, 10, 1000))
_bandpass = signal.butter(2, (10 / 500, 350 / 500), btype="band", output="sos")
_FILTER = np.vstack((_bandpass, _notch))


def prefilter(data):
    """Prefilter data with offset removal and 60 Hz notch filter.

    Args:
        data (np.ndarray): Data of shape (..., N_samples, N_channels)

    Returns:
        np.ndarray: filtered data
    """
    return signal.sosfilt(_FILTER, data, axis=-2)


def analyze(data, percentile=98) -> tuple:
    """
    Return rectified max, percentile,  median, MAV, STD of data, along its columns.
    """
    adata = np.abs(data)
    max = np.max(adata)
    ninefive = np.percentile(adata, percentile)
    mean = np.mean(adata)
    median = np.median(adata)
    std = np.mean(np.std(data, 0))

    return max, ninefive, median, mean, std


def twos_comp(val: np.ndarray, bits: int) -> np.ndarray:
    # Stolen from https://stackoverflow.com/questions/1604464/twos-complement-in-python
    """Convert a `bits`-long uint to a signed integer."""
    val = (val & (1 << bits) - 1).astype(int)  # clear bits
    return val - ((val & (1 << (bits - 1))) << 1)  # two's complement


def echo_c(data, *kargs):
    return data


def echo_d(data, *kargs):
    return data


def run_test(
    compress_method, reconstruction_method, bits, dataset, dataset_root: str, **kwargs
):
    """Run test.

    Args:
        compress_method (Callable): Callable which takes in `np.ndarray`, `int` arguments and returns `np.ndarray`
        reconstruction_method (Callable): Callable which in `np.ndarray`, `int` arguments and returns `np.ndarray`
        bits (int): Number of bits to compress to
        dataset: either "EMAGER" or "capgmyo"
        position (int, optional): Position of the progress bar. Defaults to 0.
    """
    results = []
    iter_fn = iter_capgmyo if dataset == "capgmyo" else iter_emager
    dataset_path = dataset_root + dataset

    for x in tqdm(iter_fn(dataset_path), **kwargs):
        # Iterates over repetitions of data
        # x has shape (N, C)
        # if len(results) == 10:
        #     break
        xq = compress_method(x, bits)
        xbar = reconstruction_method(xq, bits)
        ae = np.abs(x[: len(xbar)] - xbar)
        results.append(ae)
    return results


def save_results(results, filename):
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)


def iter_capgmyo(dataset_root: str, preprocess=True):
    """Iterate on CapgMyo dataset, returns data in millivolts with shape (1000, 128)"""
    if not dataset_root.endswith("/"):
        dataset_root += "/"
    for f in os.listdir(dataset_root):
        yield loadmat(dataset_root + "/" + f)["data"]


def load_capgmyo_dbb(dataset_root: str, analog: bool = False):
    """Load CapgMyo dataset, returns data in millivolts with shape (N, 128)"""
    if not dataset_root.endswith("/"):
        dataset_root += "/"
    data = []
    for subject in range(1, 11):
        for session in range(1, 3):
            folder = dataset_root + f"subject{subject:02d}_session{session}/"
            files = os.listdir(folder)
            files.sort()
            for f in files:
                mat = loadmat(folder + f)
                d = mat["data"]
                data.append(d)

    data = np.array(data).reshape(-1, 128)
    if not analog:
        # 16-bit acquisitions from [-2.5, 2.5] mV
        data = 2**15 * data / 2.5
    return data.astype(np.float32)


def load_capgmyo_dba(dataset_root: str, analog: bool = False):
    """Load CapgMyo dataset, returns data in millivolts with shape (N, 128)"""
    if not dataset_root.endswith("/"):
        dataset_root += "/"
    data = []
    for f in os.listdir(dataset_root):
        data.append(loadmat(dataset_root + f)["data"])
    data = np.array(data).reshape(-1, 128)
    if not analog:
        # 16-bit acquisitions from [-2.5, 2.5] mV
        data = (2**15 * data / 2.5).astype(np.int16)
    return data


if __name__ == "__main__":
    # Example usage
    dataset_root = "/home/gabrielgagne/Documents/Datasets/capgmyo"
    data = None
    for d in iter_capgmyo(dataset_root):
        # print(data.shape)
        if data is None:
            data = d
        else:
            data = np.concatenate((data, d), axis=0)
    print(data.shape)
    print(analyze(data))
