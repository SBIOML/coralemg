import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import dataset_definition as dtdef
from sklearn.metrics import ConfusionMatrixDisplay
from scipy import stats
import seaborn as sns 

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
    dataset,
    result_path,
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

    dataset_name = dataset.name

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

    return global_accuracy, global_accuracy_maj

def display_confusion_matrix(
    dataset,
    result_path,
    model_name,
    subjects,
    sessions,
    compression_method,
    bit,
    fine_tuned=False,
    ondevice=False,
):
    dataset_name = dataset.name
    nb_class = dataset.nb_class

    cm = np.zeros((nb_class, nb_class))
    cm_maj = np.zeros((nb_class, nb_class))
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
            conf_matrix = data["confusion_matrix"]
            conf_matrix_maj = data["confusion_matrix_maj"]
            for i in range(5):
                cm += conf_matrix[i]
                cm_maj += conf_matrix_maj[i]

    #display confusion matrix
    cm = 100*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ConfusionMatrixDisplay(cm).plot()

    cm_maj = 100*cm_maj.astype('float') / cm_maj.sum(axis=1)[:, np.newaxis]
    ConfusionMatrixDisplay(cm_maj).plot()
    plt.show()

def evaluate_time(
    dataset,
    result_path,
    model_name,
    subjects,
    sessions,
    compression_methods,
    bits,
    fine_tuned=False,
    ondevice=False,
):
    dataset_name = dataset.name

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
            nb_bins = 2**8

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
    #plt.savefig("histogram_quant.png")
    plt.show()

def evaluate_repartition_relative(dataset_path, subjects, sessions, compressed_methods, bit):
    nb_bins = 2**bit

    color_map = {
        "baseline": "#1f77b4",     # blue
        "minmax": "#ff7f0e",       # green orange
        "msb": "#2ca02c",   # orange green
        "smart": "#d62728",      # red
        "root": "#9467bd",       # purple
    }

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
            nb_bins = 2**bit
            
        data_array_pct = 100 * data_array / (nb_bins - 1)

        plt.hist(
            data_array_pct,
            bins=nb_bins,
            range=(0, 100),
            alpha=0.5,
            rwidth=1.0,
            weights=np.ones_like(data_array_pct)*100 / len(data_array_pct),
            color=color_map.get(compression_method.lower(), "gray")
        )
        plt.ylim((0, 20))
        plt.xlim((-0.1, 60))
        plt.xlabel("Relative Amplitude [%]", fontsize=14)
        plt.ylabel("Density [%]", fontsize=14)
        plt.grid(alpha=0.3)
    plt.legend(["Original","Min-Max", "Right Shift", "Smart-3", "Root-3"])
    plt.savefig("histogram_quant_%sbits.png"%(bit))
    plt.show()

def mean_hypothesis_test(dataset,
    result_path,
    model_name,
    subjects,
    sessions,
    compression_method,
    bit,
    fine_tuned,
    ondevice,
):
    baseline_accuracy_maj = np.array([])
    _, quantized_accuracy_maj = evaluate_accuracy(dataset, result_path, model_name, subjects, sessions, compression_method, bit, fine_tuned, ondevice)
    for i in range(8):
        _, ba_acc_maj = evaluate_accuracy(dataset, result_path, model_name, subjects, sessions, "baseline", i+1, fine_tuned, ondevice)
        baseline_accuracy_maj = np.append(baseline_accuracy_maj, ba_acc_maj)
    
    baseline_accuracy_maj = baseline_accuracy_maj.flatten()

    u_stat, p_value = stats.mannwhitneyu(quantized_accuracy_maj, baseline_accuracy_maj, alternative="greater")
    print("p-value:", p_value)

def draw_box_plot(dataset, 
    project_path, 
    model_name, 
    subjects,
    bit,
):
    medianprops = dict(linestyle='-', linewidth=2, color='black')
    meanpointprops = dict(marker='o', markeredgecolor='black',
                        markerfacecolor='darkblue')
    boxprops=dict(linewidth=2, alpha=0.9)
    whiskerprops=dict(linewidth=2, linestyle=':')
    capprops=dict(linewidth=2)
    fontsize=15

    #Create a dataframe with subject, model, roll, accuracy
    dataframe = pd.DataFrame(columns=['subject', 'session', 'compression', 'fine-tuning', 'vote', 'accuracy'])

    compressions = ["root", "smart", "minmax", "msb", "baseline"]
    fine_tunings = ["ondevice", "finetuned", ""]

    for compression in compressions:
        for fine_tuning in fine_tunings:
            dataframe = _generate_results_list(dataframe, dataset, project_path, model_name, subjects, compression, bit, fine_tuning)

    #Keep only the vote results
    dataframe_novote = dataframe[dataframe['vote'] == "no_vote"]
    dataframe_vote = dataframe[dataframe['vote'] == "vote"]

    # Draw box plot
    fig = plt.figure(figsize=(14,8))
    sns.boxplot(data=dataframe_vote, x="fine-tuning", y="accuracy", hue="compression", gap=0.2,
                medianprops=medianprops, 
                meanprops=meanpointprops, 
                palette=["paleturquoise","navajowhite","#c1b7ff", "lightsalmon", "mediumseagreen"],
                showmeans=True,
                boxprops=boxprops, 
                whiskerprops=whiskerprops, 
                capprops=capprops,
                linecolor='black',
                width=0.65)
    #plt.title("Jeu de données : %s - Vote de majorité : 25 ms" % (dataset_name), fontsize=17)
    plt.tick_params(axis='both', which='major', labelsize=19)
    plt.legend(title='Quantization: %s-bit'%(bit), fontsize=16, title_fontsize=16, loc='lower right')
    plt.xlabel("")
    plt.ylabel("Accuracy (%)", fontsize=19)
    plt.yticks([10*i for i in range(11)])
    plt.grid(axis='y')
    plt.ylim(0, 105)
    fig.tight_layout()
    fig.savefig('result_tpu_quant_%sbits_%s.png'%(bit, dataset.name), transparent=True)
    plt.show()

def _generate_results_list(dataframe, dataset, project_path, model_name, subjects, compression_method, bit, tuning=""):
    sessions = ["1", "2"]
    dataset_name = dataset.name
    nb_no_votes = 1
    nb_votes = 6 

    compression_dict = {"baseline":"Baseline", "root":"Root", "smart":"Smart", "minmax":"Min-Max", "msb":"Right"}

    tuning_load = "" if len(tuning) == 0 else "_%s"%(tuning)
    
    if tuning == "":
        tuning_name = "No tuning"
    elif tuning == "finetuned":
        tuning_name = "Off-Device"
    elif tuning == "ondevice":
        tuning_name = "On-Device"

    for subject in subjects:
        for session in sessions:
            if compression_method == "baseline":
                for index in range(8):
                    ok = True
                    curr_bit = index+1
                    if tuning != "ondevice":
                        datapath = "%s/offdevice_results/%s_%s_%s_%s_%s_%sbits_evaluation%s.npz"%(project_path, dataset_name, model_name, subject, session, compression_method, curr_bit, tuning_load)
                    else:
                        ok = False
                    if ok:
                        data = np.load(datapath)

                        no_vote = 100*np.mean(data["accuracy"])
                        vote = 100*np.mean(data["accuracy_majority_vote"])

                        vote_name = "no_vote"
                        liste = [subject, session, compression_dict[compression_method], tuning_name, vote_name, no_vote]
                        dataframe = pd.concat([pd.DataFrame([liste], columns=dataframe.columns), dataframe], ignore_index=True)

                        vote_name = "vote"
                        liste = [subject, session, compression_dict[compression_method], tuning_name, vote_name, vote]
                        dataframe = pd.concat([pd.DataFrame([liste], columns=dataframe.columns), dataframe], ignore_index=True)

            else:
                datapath = "%s/ondevice_results/%s_%s_%s_%s_%s_%sbits_evaluation%s.npz"%(project_path, dataset_name, model_name, subject, session, compression_method, bit, tuning_load)
                data = np.load(datapath)

                no_vote = 100*np.mean(data["accuracy"])
                vote = 100*np.mean(data["accuracy_majority_vote"])

                vote_name = "no_vote"
                liste = [subject, session, compression_dict[compression_method], tuning_name, vote_name, no_vote]
                dataframe = pd.concat([pd.DataFrame([liste], columns=dataframe.columns), dataframe], ignore_index=True)

                vote_name = "vote"
                liste = [subject, session, compression_dict[compression_method], tuning_name, vote_name, vote]
                dataframe = pd.concat([pd.DataFrame([liste], columns=dataframe.columns), dataframe], ignore_index=True)

    return dataframe

if __name__ == "__main__":

    subjects = ["00","01","02","03","04","05","06","07","08","09","10","11"]

    #subjects = ["01","02","03","04","05","06","07","08","09","10"]

    sessions = ["1", "2"]
    #compression_methods = ["minmax", "msb", "smart", "root"]

    project_path = "offdevice"
    result_path = "offdevice/ondevice_results"
    #result_path = "offdevice/offdevice_results"
    dataset = dtdef.EmagerDataset()
    #dataset = dtdef.CapgmyoDataset()
    model_name = "cnn"

    # bits = [1,2,3,4,5,6,7,8]
    # for bit in bits :
    #     evaluate_accuracy(
    #         dataset, result_path, model_name, subjects, sessions, "smart", bit, fine_tuned=False, ondevice=False
    #     )
    
    # mean_hypothesis_test(dataset, result_path, model_name, subjects, sessions, "minmax", 6, fine_tuned=True, ondevice=False)
    for i in range(8):
        draw_box_plot(dataset, project_path, model_name, subjects, str(i+1))

    # evaluate_time(
    #     dataset,
    #     result_path,
    #     subjects,
    #     sessions,
    #     compression_methods,
    #     fine_tuned=False,
    #     ondevice=True,
    # )

    #compression_methods = ["minmax", "msb", "smart", "root"]
    #dataset_path = "dataset/train/capgmyo/"
    #sessions = ["1", "2"]
    #evaluate_repartition_relative(dataset_path, subjects, sessions, compression_methods, 8)
