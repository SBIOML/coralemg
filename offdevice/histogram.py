import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import utils
from data_processing import preprocess_data, compress_data


def plot_emg_histogram(
    emg_data,
    bins: int | np.ndarray = 50,
    xlabel="Relative Amplitude (%)",
    ylabel="Density (%)",
    color="skyblue",
    alpha=0.7,
    figsize=(10, 6),
    xlim=(0, 60),
    ylim=(0, 20),
    fig=None,
    ax=None,
):
    """
    Plot a histogram of EMG (electromyography) data.

    Parameters:
    -----------
    emg_data : numpy.ndarray
        1D array containing EMG signal data
    bins : int, default=50
        Number of histogram bins
    title : str, default='EMG Data Histogram'
        Plot title
    xlabel : str, default='Amplitude (μV)'
        X-axis label
    ylabel : str, default='Frequency'
        Y-axis label
    color : str, default='skyblue'
        Histogram color
    alpha : float, default=0.7
        Transparency level (0-1)
    figsize : tuple, default=(10, 6)
        Figure size (width, height)

    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """

    # Ensure input is a numpy array
    if not isinstance(emg_data, np.ndarray):
        emg_data = np.array(emg_data)

    # Flatten array if multidimensional
    emg_data = emg_data.flatten()

    # Remove any NaN or infinite values
    emg_data = emg_data[np.isfinite(emg_data)]

    # Create the plot
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    # Plot histogram
    hist, edges = np.histogram(emg_data, bins=bins)
    hist = 100 * hist / len(emg_data)
    ax.bar(
        bins[:-1],
        hist,
        width=np.diff(bins),
        color=color,
        alpha=alpha,
        # edgecolor="black",
        linewidth=0.5,
        align="edge",
        # rwidth=1.0,
    )

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # Add statistics as text
    # mean_val = np.mean(emg_data)
    # std_val = np.std(emg_data)
    # stats_text = (
    #     f"Mean: {mean_val:.2f} mV\nStd: {std_val:.2f} mV\nSamples: {len(emg_data)}"
    # )
    # ax.text(
    #     0.02,
    #     0.98,
    #     stats_text,
    #     transform=ax.transAxes,
    #     fontsize=10,
    #     verticalalignment="top",
    #     bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    # )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    return fig, ax


if __name__ == "__main__":
    matplotlib.rcParams.update({"font.size": 18})

    DATASET_ROOT = "../../Documents/Datasets/"

    bitwidths = [1, 2, 3, 4, 5, 6, 7, 8]
    # bitwidths = [8]
    # bitwidths.reverse()

    quant_methods = [
        "minmax",
        "msb",
        "smart",
        "root",
    ]
    # colors = ["orange", "green", "red", "purple", "brown", "pink", "gray"]
    colors = [
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
    ]
    datasets = ["EMAGER", "CAPGDBB"]
    # datasets = ["EMAGER"]
    # datasets = ["CAPGDBB"]

    for dataset in datasets:
        if dataset == "EMAGER":
            data = utils.load_emager(DATASET_ROOT + "EMAGER", True).reshape(-1, 25, 64)
        else:
            data = utils.load_capgmyo_dbb(DATASET_ROOT + "CAPGDBB").reshape(-1, 25, 128)

        data = data - np.mean(data, axis=1, keepdims=True)  # DC removal
        data = np.abs(data)  # absolute value
        data = np.mean(data, axis=1)  # average across windows

        # fig, ax = plot_emg_histogram(
        #     100 * (data / 2**15), bins=np.arange(0, 100.1, 0.1), alpha=0.5, xlim=(0, 5)
        # )
        plt.tight_layout()
        # plt.savefig(f"out/emg_histogram_{dataset}_original.png", dpi=300)

        for b in bitwidths:
            incr = 100 / 2**b
            bins = np.arange(0, 100 + incr, incr)
            fig, ax = plot_emg_histogram(
                100 * (data / 2**15), color="#1f77b4", bins=bins, alpha=0.5
            )
            if b == 8:
                xlim = (0, 60)
                ylim = (0, 20)
            else:
                xlim = (0, 100)
                ylim = (0, 100)

            for qm, color in zip(quant_methods, colors):
                # Apply quantization method
                quantized_data = compress_data(data, qm, b).astype(np.float32)
                quantized_data = 100 * quantized_data / 2**b
                fig, ax = plot_emg_histogram(
                    quantized_data,
                    bins=bins,
                    fig=fig,
                    ax=ax,
                    color=color,
                    alpha=0.5,
                    xlim=xlim,
                    ylim=ylim,
                )
            ax.legend(
                [
                    "Original",
                    "Min-max",
                    "Right shift",
                    "Smart ($ m=3 $)",
                    "Root ($ r=3.0 $)",
                ]
            )
            plt.tight_layout()
            folder = "coralemg" if dataset == "EMAGER" else "capgmyo"
            ds_name = "emager" if dataset == "EMAGER" else "capgmyo"
            plt.savefig(
                f"figures/{folder}/histogram_quant_{b}bits_{ds_name}.png", dpi=300
            )
            print("Saved figure for", ds_name, "with", b, "bits.")
    # plt.show()
