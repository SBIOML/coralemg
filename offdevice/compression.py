import numpy as np
import save_dataset as sd
import data_processing as dp
import pandas as pd


def get_csv(root):
    """
    Find all csv files under a directory (recursive).
    They can be loaded with numpy.genfromtxt(..., delimiter=',')
    """
    res = []
    for dir_path, _, file_names in os.walk(root):
        res.extend([f"{dir_path}/{f}" for f in file_names])

    return res


def analyze(data, axis=None):
    """
    Return rectified max, 98th prct, median, MAE, STD of data, along its columns.
    """
    adata = np.abs(data)
    max = np.max(adata, axis)
    ninefive = np.percentile(adata, 98, axis)
    mean = np.mean(adata, axis)
    median = np.median(adata, axis)

    std = (
        np.std(data)
        if len(data.shape) == 1 or data.shape[1] == 1
        else np.mean(np.std(data, 1), axis)
    )

    return max, ninefive, median, mean, std


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


# -----------
# Analysis functions, not necessarily flexible used standalone
# -----------


def histogram(data_df):
    """
    Params:
        - data_df is a (n, 1) array
    """
    # histogram of signals for each gesture
    print(analyze(data_df))
    # data_df *= 255.0 / 32768.0
    data_df.hist(density=True, bins=10000, rwidth=1.0, alpha=1.0)
    # plt.title("Histogram of 25-ms averaged blocks of processed data")
    plt.title("")
    plt.ylabel("Density", fontsize=14)
    plt.xlabel("Amplitude", fontsize=14)
    plt.grid(alpha=0.3)
    plt.minorticks_on()
    plt.xlim((0, 1200))
    plt.savefig("histogram_proc.png")
    plt.show()
    # dhist, _ = np.histogram(data_df, bins)

    """rem = 100.0
    for i, dh in enumerate(dhist):
        prct = 100 * dh / data_df.size
        rem -= prct
        print(
            f"Bin {bins[i]:.0f}-{bins[i+1]:.0f} : {prct:.5f}%, {rem:.4f}% of samples remaining"
        )
    print("----------")
    """


def concat_gestures(path):
    subjects = [f"{i:03d}" for i in range(12)]
    session = ["001", "002"]
    concat = [np.ndarray((0, 64)) for _ in range(6)]
    for sub in subjects:
        for ses in session:
            print(f"Processing subject {sub} session {ses}")
            try:
                data_array = sd.getData_EMG(path, sub, ses, differential=False)
                procd = dp.preprocess_data(data_array)
                for i in range(6):
                    shaped = np.reshape(procd[i], (procd.shape[1] * procd.shape[2], 64))
                    concat[i] = np.vstack([concat[i], shaped])
                    # print(concat[i].shape)
            except FileNotFoundError:
                continue
    return concat


def load_concat_gestures():
    ret = []
    for i in range(6):
        with open(f"gesture_concat_{i}.npy", "rb") as f:
            ret.append(np.load(f))
    return ret


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import os

    # Concatenate all training data
    dataset_path = f"{os.environ['HOME']}/Documents/Datasets/EMAGER/"
    print(dataset_path)
    data_concat = []
    if "gesture_concat_5.npy" not in os.listdir():
        data_concat = concat_gestures(dataset_path)
        for i, g in enumerate(data_concat):
            with open(f"gesture_concat_{i}.npy", "wb") as f:
                np.save(f, g)
    else:
        data_concat = load_concat_gestures()

    # Serialize training data
    data_c = np.ndarray((0, 1))
    for i in range(6):
        data_c = np.vstack(
            (
                data_c,
                np.reshape(
                    data_concat[i],
                    (data_concat[i].shape[0] * data_concat[i].shape[1], 1),
                ),
            )
        )

    print(f"Expected dataset length: {13*2*10*5000*6/25} samples")
    print(f"Actual dataset length: {data_c.size/64} samples")

    print(np.min(data_c))
    print(np.max(data_c))
    print(np.mean(data_c))
    hist_dict = {"minmax": np.reshape(data_c, (-1,)) * 255.0 / 32767}

    data_df = pd.DataFrame(data_c)
    histogram(data_df)

    # Show compression comparison
    error_list = []

    # Test quantization methods

    # Smart bitshift
    for i in [1, 2, 3, 4]:
        cgesture = smart_bitshift_c(np.round(data_c).astype(np.uint16), 8, i)
        cdgesture = smart_bitshift_d(cgesture, 8, i)
        error = cdgesture - data_c
        error_list.append([i, *[f"{e:.02f}" for e in analyze(error)]])
        if i == 3:
            hist_dict["smart-3"] = cgesture.reshape((-1,))

    """# Log
    for a in [10000, 15000, 16383, 20000, 25000, 32767]:
        cgesture = log_c(data_c, a)
        cdgesture = log_d(cgesture, 4, a)
        error = cdgesture - data_c
        err_df = error_list.append([a, *[f"{e:.02f}" for e in analyze(error)]])
    """

    # Root(exp)
    for exp in [2.0, 2.5, 3.0, 3.5]:
        cgesture = nroot_c(data_c, exp, 20000)
        cdgesture = nroot_d(cgesture, exp, 20000)
        error = cdgesture - data_c
        error_list.append([exp, *[f"{e:.02f}" for e in analyze(error)]])
        if exp == 3:
            hist_dict["root-3"] = cgesture.reshape((-1,))

    # Naive bitshift
    cdgesture = naive_bitshift_cd(np.round(data_c).astype(np.uint16), 8)
    error = cdgesture - data_c
    error_list.append(["naive", *[f"{e:.02f}" for e in analyze(error)]])

    hist_dict["rightshift"] = (np.round(data_c.astype(np.int16)) >> 8).reshape((-1,))

    # MinMax
    error = np.ndarray((0, 64))
    for subject in [f"{i:03d}" for i in range(12)]:
        for session in ["001", "002"]:
            raw_data = None
            proc_data = None
            data_array = sd.getData_EMG(
                dataset_path, subject, session, differential=False
            )
            averages_data = dp.preprocess_data(data_array)
            X, y = dp.extract_with_labels(averages_data)
            X_compressed = dp.compress_data(X, method="minmax")
            # print(X_compressed.shape)
            terror = X - (
                X_compressed
                * (np.max(X, 1, keepdims=True) - np.min(X, 1, keepdims=True))
                / 255.0
                + np.min(X, 1, keepdims=True)
            )
            error = np.vstack((error, terror))

    error_list.append(["minmax", *[f"{e:.02f}" for e in analyze(error)]])

    cols = ["MSB shift", "Maximum", "95th percentile", "Median", "Mean", "STD"]
    error_table = pd.DataFrame(error_list, columns=cols)
    print(error_table.to_latex(index=False))
