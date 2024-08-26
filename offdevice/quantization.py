import numpy as np
import save_dataset as sd
import data_processing as dp
import pandas as pd


def normalize_min_max_c(data, bits):
    """
    Normalize each time_sample in a (time_sample, channel) array so that the channels are in a [0, 1] range for each time_sample.
    Then return as uint8

    @param data the data to normalize
    @param bits the number of bits to compress to

    @return the normalized data

    """

    max_value = (2**bits)-1

    tmp_data = (data - np.min(data, 1, keepdims=True)) / (
        np.max(data, 1, keepdims=True) - np.min(data, 1, keepdims=True)
    )
    return np.round(tmp_data * max_value)


def naive_bitshift_cd(data, bits):
    """
    Apply right bit shifting to compress the data and restore to the original range with left shifting by the same 
    amount of bits.

    @param data the data to compress/decompress
    @param bits the number of bits compressed to
    
    @return the rescaled data

    """
    tmp_data = np.round(data).astype(np.uint16)
    shift = 16 - bits
    return (tmp_data >> shift) << shift


def naive_bitshift_c(data, bits):
    """
    Apply right bit shifting to compress the data

    @param data the data to compress
    @param bits the number of bits compressed to
    
    @return the compressed data

    """
    tmp_data = np.round(data).astype(np.uint16)
    shift = 16 - bits
    return tmp_data >> shift


def smart_bitshift_c(data, bits, msb_shift):
    """
    Apply bit shifting to compress the data cutting the specified lsb_bits and msb_bits

    @param data the data to compress
    @param bits the number of bits compressed to
    @param msb_bits the number of msb_bits used for the left shifting
    
    @return the compressed data

    """
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
    """
    
    """
    return data.astype(np.uint16) << (16 - bits - msb_shift)


def log_c(data, atg, bits):
    """
    Quantize `data` into its logarithm-`base` representation, and normalizes to 0-(2**bits)-1 by doing:

    result*(2**bits)-1/atg,

    @param data the data to compress
    @param atg Andre The Giant value of average EMG activation aka ceiling to normalize against
    @param bits the number of bits to compress to

    @return the log of data scaled to [0,(2**bits)-1]
    """

    max_value = float((2**bits)-1)

    log_data = max_value * np.emath.logn(atg, data)
    return np.clip(np.round(log_data), 0, max_value)


def log_d(data, base, atg, bits):

    max_value = float((2**bits)-1)

    atgl = np.emath.logn(base, atg)
    return np.power(base, atgl * data.astype(np.float64) / max_value)


# TODO : max relative to data (max = np.max(data) or global ?
def nroot_c(data, exp, max, bits):
    """
    Quantize `data` into its `exp`th-root representation, and normalizes the result to 0-(2**bits)-1.

    Max is used as the scaling ceiling.

    Returns the log of data scaled to [0,(2**bits)-1]
    """

    max_value = (2**bits)-1

    rt_data = np.clip(np.round(np.power(data / max, 1 / exp) * max_value), 0, max_value)
    return np.round(rt_data)


def nroot_d(data, exp, max, bits):
    max_value = float((2**bits)-1)

    return np.power(data.astype(np.float64) / max_value, exp) * max


