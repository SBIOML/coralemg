# CORALEMG

This repository contains the source code used in the _TinyML for Real-Time Embedded HD-EMG Hand
Gesture Recognition with On-Device Fine-Tuning_ paper. As described in detail in the paper, it includes the server-side (Host PC) as well as embedded device-side (Google Coral Mini) code.

![Alt Text](https://github.com/SBIOML/coralemg/blob/main/coralemg_realtime.gif)

## Set up

First, clone this repository in a convenient working directory, and run: `python3 -m venv .; source bin/activate; python3 -m pip install -r requirements.txt` to install the dependencies.

Afterwards, it's recommended to download the CoralEMG dataset from [this Kaggle link](https://www.kaggle.com/datasets/etiennebuteau/coralemg) and extract it to your working directory, giving the directory structure `<coralemg>/dataset/emager/`.

## Using this repository

### Adding quantization methods

Quantization methods are defined in `offdevice/quantization.py` and/or `ondevice/quantization.py`. In general, functions should take in a `numpy` data array and return the compressed data.

Then, add the method(s) in `data_processing.py`'s `compress_data` function with the appropriate arguments. As seen in `offdevice/training_model.py`, you can then create a list with the desired quantization methods and pass it to `train_all_subjects` or `finetune_all_subjects`.

If you want to analyze the performance of quantization methods, you can use `offdevice/analyze_data.py`'s functions.

### Preprocess CoralEMG dataset

Run `python3 offdevice/save_dataset.py`. Processed and quantized CoralEMG will be saved in `dataset/train/<quantization_method>`. You can also add or remove quantization methods to be exported.

### Off-device training

Run `python3 offdevice/training_model.py`. The Tensorflow models will be created and trained for every quantization method. They are saved to `model/emager_<subject>_<session>_<quant>.h5`.

### Off-device evaluation

Run `python3 offdevice/evaluate_model.py`. Every pre-exported Tensorflow models will be evaluated. The results are printed to the terminal and are also saved into `offdevice_results/` as Numpy archives with keys: accuracy, accuracy_majority_vote, confusion_matrix and confusion_matrix_maj. 

Modify and run `python3 offdevice/analyze_data.py` to get key statistics from the selected users and compression method. 

### Compiling models for the edge TPU

The [Edge TPU Compiler](https://coral.ai/docs/edgetpu/compiler/#system-requirements) must be first installed. 

Run `python3 offdevice/convert_to_tflite.py` to convert the tensorflow models to tensorflow-lite.
In each model/tflite/(normal,tuned,extractor) subdirectory, a compile_toedge.py script can be run directly from the subdirectory to compile the models for the Edge TPU.

### On-device training

Guide in progress

### On-device evaluation

The results are saved into `ondevice_results/` as Numpy archives with keys: accuracy, accuracy_majority_vote, confusion_matrix, confusion_matrix_maj, inference_time, maj_vote_time and process_time.

### End-to-end system

The end-to-end system needs code to run both on the _host_ and _device_, which act as _server_ and _client_, respectively. Communication and file transfers are ensured via SSH.

#### Off-device

Run `python3 offdevice/server.py`. Before starting it, you should configure the IP address of your Coral device with a text file located at `config/ip_addrs/coral_ip.txt`. First, it opens an internet socket on port 6677.

#### On-device

## Citing

If you find this work helpful and want to use it or refer to it, please consider citing it with the following:

To complete after publication

    @article{buteau2024coralemg,
        title={TinyML for Real-Time Embedded HD-EMG Hand
        Gesture Recognition with On-Device Fine-Tuning},
        author={Etienne Buteau, Gabriel Gagné, William Bonilla, Mounir Boukadoum, Paul Fortier and Benoit Gosselin}
    }
