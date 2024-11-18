import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def histogram(data_df: pd.DataFrame, xlim=1200):
    """
    Show an histogram from a pandas DataFrame's keys.

    Params:
        - data_df the DataFrame to create the histogram from
        - xlim : histogram's maximum x-axis value

    Returns nothing, but shows the `plt` histogram
    """
    data_df.hist(density=True, bins=10000, rwidth=1.0, alpha=1.0)
    plt.title("")
    plt.ylabel("Density", fontsize=14)
    plt.xlabel("Amplitude", fontsize=14)
    plt.grid(alpha=0.3)
    plt.minorticks_on()
    plt.xlim((0, xlim))
    plt.savefig("histogram_proc.png")
    plt.show()
    # dhist, _ = np.histogram(data_df, bins)


def analyze(data, axis=None, centile=98):
    """
    Get key statistics from a data array.

    Parameters:
        - data : data array to analyze
        - axis : axis along which to calculate the statistics
        - centile : percentile to calculate

    Return rectified max, 98th prct, median, MAE, STD of data, along its columns.
    """
    adata = np.abs(data)
    max = np.max(adata, axis)
    ninefive = np.percentile(adata, centile, axis)
    mean = np.mean(adata, axis)
    median = np.median(adata, axis)

    std = (
        np.std(data)
        if len(data.shape) == 1 or data.shape[1] == 1
        else np.mean(np.std(data, 1), axis)
    )

    return max, ninefive, median, mean, std


def evaluate_accuracy(
    result_path,
    dataset_name,
    model_name,
    subjects,
    sessions,
    compression_method,
    bit,
    fine_tuned=False,
    ondevice=False,
):
    global_accuracy = np.array([])
    global_accuracy_maj = np.array([])

    #TODO Fix
    cm = np.zeros((6, 6))
    cm_maj = np.zeros((6, 6))
    for subject in subjects:
        for session in sessions:
            if fine_tuned:
                datapath = "%s/%s_%s_%s_%s_%s_%sbits_evaluation_finetuned.npz" % (
                    result_path,
                    dataset_name,
                    model_name,
                    subject,
                    session,
                    compression_method,
                    bit,
                )
            elif ondevice:
                datapath = "%s/%s_%s_%s_%s_%s_%sbits_evaluation_ondevice.npz" % (
                    result_path,
                    dataset_name,
                    model_name,
                    subject,
                    session,
                    compression_method,
                    bit,
                )
            else:
                datapath = "%s/%s_%s_%s_%s_%s_%sbits_evaluation.npz" % (
                    result_path,
                    dataset_name,
                    model_name,
                    subject,
                    session,
                    compression_method,
                    bit,
                )

            data = np.load(datapath)
            global_accuracy = np.append(global_accuracy, data["accuracy"])
            global_accuracy_maj = np.append(
                global_accuracy_maj, data["accuracy_majority_vote"]
            )
            conf_matrix = data["confusion_matrix"]
            conf_matrix_maj = data["confusion_matrix_maj"]
            for i in range(5):
                cm += conf_matrix[i]
                cm_maj += conf_matrix_maj[i]

    print("Dataset : %s" % (dataset_name))
    print("Compressed method: %s" % (compression_method))
    print("Number of bits: %s" % (bit))
    print("Accuracy")
    print(
        "{:.2f}".format(100 * np.mean(global_accuracy)),
        "+-",
        "{:.2f}".format(100 * np.std(global_accuracy)),
    )

    print("Accuracy majority vote")
    print(
        "{:.2f}".format(100 * np.mean(global_accuracy_maj)),
        "+-",
        "{:.2f}".format(100 * np.std(global_accuracy_maj), "\n"),
    )
    print("\n")

    # #display confusion matrix
    # cm = 100*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # ConfusionMatrixDisplay(cm).plot()

    # cm_maj = 100*cm_maj.astype('float') / cm_maj.sum(axis=1)[:, np.newaxis]
    # ConfusionMatrixDisplay(cm_maj).plot()
    # plt.show()


def evaluate_time(
    result_path,
    dataset_name,
    model_name,
    subjects,
    sessions,
    compression_methods,
    bits,
    fine_tuned=False,
    ondevice=False,
):
    if "offdevice" in result_path:
        print("Offdevice results not compatible with time evaluation")
        return

    inference_time = np.array([])
    majority_vite_time = np.array([])
    total_inference_time = np.array([])

    for subject in subjects:
        for session in sessions:
            for compression_method in compression_methods:
                for bit in bits:
                    if fine_tuned:
                        datapath = "%s/%s_%s_%s_%s_%s_%sbits_evaluation_finetuned.npz" % (
                            result_path,
                            dataset_name,
                            model_name,
                            subject,
                            session,
                            compression_method,
                            bit
                        )
                    elif ondevice:
                        datapath = "%s/%s_%s_%s_%s_%s_%sbits_evaluation_ondevice.npz" % (
                            result_path,
                            dataset_name,
                            model_name,
                            subject,
                            session,
                            compression_method,
                            bit
                        )
                    else:
                        datapath = "%s/%s_%s_%s_%s_%s_%sbits_evaluation.npz" % (
                            result_path,
                            dataset_name,
                            model_name,
                            subject,
                            session,
                            compression_method,
                            bit
                        )

                data = np.load(datapath)

                inf_time = data["inference_time"]
                # Flatten the inference time array
                inf_time = np.array([item for sublist in inf_time for item in sublist])
                inference_time = np.append(inference_time, inf_time)

                maj_vote_time = data["maj_vote_time"]
                # Substract each point with it's previous point to get the time between each majority vote
                max_index = np.argmax(maj_vote_time, axis=1)
                maj_vote_time = np.delete(maj_vote_time, max_index, axis=1)
                maj_vote_time = np.array(
                    [item for sublist in maj_vote_time for item in sublist]
                )
                majority_vite_time = np.append(majority_vite_time, maj_vote_time)

                process_time = data["process_time"]
                process_time = np.array(
                    [item for sublist in process_time for item in sublist]
                )
                # Remove all value smaller than 3ms
                total_inference_time = np.append(total_inference_time, process_time)

    print("Median Inference Time")
    print("{:.2f}".format(1000 * np.median(inference_time)), "ms")
    print("99.5th percentile Inference Time")
    print("{:.2f}".format(1000 * np.percentile(inference_time, 99.5)), "ms")
    print("Max Inference Time")
    print("{:.2f}".format(1000 * np.max(inference_time)), "ms")
    print("\n")

    print("Median Process Time")
    print("{:.2f}".format(1000 * np.median(total_inference_time)), "ms")
    print("99.5th percentile Process Time")
    print("{:.2f}".format(1000 * np.percentile(total_inference_time, 99.5)), "ms")
    print("Max Process Time")
    print("{:.2f}".format(1000 * np.max(total_inference_time)), "ms")
    print("\n")

    print("Median Majority Vote Time")
    print("{:.2f}".format(1000 * np.median(majority_vite_time)), "ms")
    print("99.5th percentile Majority Vote Time")
    print("{:.2f}".format(1000 * np.percentile(majority_vite_time, 99.5)), "ms")
    print("Max Majority Vote Time")
    print("{:.2f}".format(1000 * np.max(majority_vite_time)), "ms")
    print("\n\n")


def evaluate_repartition(dataset_path, subjects, sessions, compressed_methods, bit):
    nb_bins = 2**bit

    for compression_method in compressed_methods:
        data_list = []
        for subject in subjects:
            for session in sessions:
                datapath = dataset_path + "%s/%s_%s_%s_%sbits.npz" % (
                    compression_method,
                    subject,
                    session,
                    compression_method,
                    bit
                )
                with np.load(datapath) as data:
                    curr_data = data["data"]
                data_list.append(curr_data)
        data_array = np.array(data_list)
        data_array = data_array.flatten()

        if compression_method == "baseline":
            data_array = float(nb_bins-1) * data_array / 32767

        plt.hist(
            data_array,
            bins=nb_bins,
            range=(0, nb_bins-1),
            alpha=0.5,
            rwidth=1.0,
            density=True,
        )
        plt.ylim((0, 0.2))
        plt.xlim((-1, 150))
        plt.xlabel("Amplitude", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.grid(alpha=0.3)
    plt.legend(["Original", "Min-Max", "Right Shift", "Smart-3", "Root-3"])
    plt.savefig("histogram_quant.png")
    plt.show()


if __name__ == "__main__":

    #subjects = ["00","01","02","03","04","05","06","07","08","09","10","11"]

    subjects = ["01","02","03","04","05","06","07","08","09","10"]

    sessions = ["1", "2"]
    #compression_methods = ["minmax", "msb", "smart", "root"]

    # #result_path = "offdevice/ondevice_results"
    result_path = "offdevice/offdevice_results"
    dataset_name = "capgmyo"
    model_name = "cnn"

    bits = [3,4,5,6,7,8]
    for bit in bits :
        evaluate_accuracy(
            result_path, dataset_name, model_name, subjects, sessions, "root", bit, fine_tuned=False, ondevice=False
        )
    # evaluate_time(
    #     result_path,
    #     subjects,
    #     sessions,
    #     compression_methods,
    #     fine_tuned=False,
    #     ondevice=True,
    # )

    # compression_methods = ["baseline", "minmax", "msb", "smart", "root"]
    # dataset_path = "dataset/train/"
    # sessions = ["001", "002"]
    # evaluate_repartition(dataset_path, subjects, sessions, compression_methods, 7)
