import numpy as np
import save_dataset as sd
import data_processing as dp
import pandas as pd


def normalize_min_max_c(data):
    """
    Normalize each time_sample in a (time_sample, channel) array so that the channels are in a [0, 1] range for each time_sample.
    Then return as uint8

    @param data the data to normalize

    @return the normalized data

    """
    tmp_data = (data - np.min(data, 1, keepdims=True)) / (
        np.max(data, 1, keepdims=True) - np.min(data, 1, keepdims=True)
    )
    return np.round(tmp_data * 255)


def naive_bitshift_cd(data, bits):
    tmp_data = np.round(data).astype(np.uint16)
    shift = 16 - bits
    return (tmp_data >> shift) << shift


def naive_bitshift_c(data, bits):
    tmp_data = np.round(data).astype(np.uint16)
    shift = 16 - bits
    return tmp_data >> shift


def smart_bitshift_c(data, bits, msb_shift):
    tmp_data = np.round(data).astype(np.uint16)
    shift = 16 - bits
    if shift >= msb_shift:
        shift -= msb_shift
        lclip = (1 << (16 - msb_shift)) - 1  # ceiling clip
        rclip = (1 << shift) - 1  # floor clip + round
        idx = tmp_data & rclip >= rclip / 2
        tmp_data[idx] = tmp_data[idx] + (rclip + 1)  # round up
        return np.clip(tmp_data, 0, lclip) >> shift
    else:
        raise ValueError("msb_shift is too high.")


def smart_bitshift_d(data, bits, msb_shift):
    return data.astype(np.uint16) << (16 - bits - msb_shift)


def log_c(data, atg):
    """
    Quantize `data` into its logarithm-`base` representation, and normalizes to 0-255 by doing:

    result*255/atg,

    ATG = Andre The Giant value of average EMG activation aka ceiling to normalize against

    Returns the log of data scaled to [0,255]
    """
    log_data = 255 * np.emath.logn(atg, data)
    return np.clip(np.round(log_data), 0, 255)


def log_d(data, base, atg):
    atgl = np.emath.logn(base, atg)
    return np.power(base, atgl * data.astype(np.float64) / 255.0)


# TODO : max relative to data (max = np.max(data) or global ?
def nroot_c(data, exp, max):
    """
    Quantize `data` into its `exp`th-root representation, and normalizes the result to 0-255.

    Max is used as the scaling ceiling.

    Returns the log of data scaled to [0,255]
    """
    rt_data = np.clip(np.round(np.power(data / max, 1 / exp) * 255), 0, 255)
    return np.round(rt_data)


def nroot_d(data, exp, max):
    return np.power(data.astype(np.float64) / 255.0, exp) * max


