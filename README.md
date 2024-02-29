# CORALEMG

This repository contains the source code used in the _TinyML for Real-Time Embedded HD-EMG Hand
Gesture Recognition with On-Device Fine-Tuning_ paper. As described in detail in the paper, it includes the server-side (Host PC) as well as embedded device-side (Google Coral Mini) code.

![Alt Text](https://github.com/SBIOML/coralemg/blob/main/coralemg_realtime.gif)

## Set up

First, clone this repository in a convenient working directory, and run: `python3 -m venv.; source bin/activate; python3 -m pip install -r requirements.txt` to install the dependencies.

Afterwards, it's recommended to download the CoralEMG dataset from [this Kaggle link](https://www.kaggle.com/datasets/etiennebuteau/coralemg) and extract it to your working directory, giving the directory structure `<coralemg>/dataset/emager/`.

## Using this repository

### Adding quantization methods

You can define quantization methods in ``

### Preprocess CoralEMG dataset

Run `python3 offdevice/save_dataset.py`. Processed and quantized CoralEMG will be saved in `dataset/train/<quantization_method>`. You can also add or remove quantization methods to be exported.

### Off-device training

Run `python3 offdevice/training_model.py`. The Tensorflow models will be created and trained for every quantization method. They are saved to `model/emager_<subject>_<session>_<quant>.h5`.

### Off-device evaluation

Run `python3 offdevice/evaluate_model.py`. Every pre-exported Tensorflow models will be evaluated. The results are printed to the terminal and are also saved into `offdevice_results/` as Numpy archives with keys: _accuracy, accuracy_majority_vote, confusion_matrix, confusion_matrix_maj_.

### On-device training

TODO

### On-device evaluation

TODO

### End-to-end system

The end-to-end system needs code to run both on the _host_ and _device_, which act as _server_ and _client_, respectively. Communication and file transfers are ensured via SSH.

#### Off-device

Run `python3 offdevice/server.py`. Before starting it, you should configure the IP address of your Coral device with a text file located at `config/ip_addrs/coral_ip.txt`. First, it opens an internet socket on port 6677.

#### On-device

## Citing

If you find this work helpful and want to use it or refer to it, please consider citing it with the following:

TODO

    @article{buteau2024coralemg,
        title={TinyML for Real-Time Embedded HD-EMG Hand
        Gesture Recognition with On-Device Fine-Tuning},
        author={Etienne Buteau, Gabriel Gagné, William Bonilla, Mounir Boukadoum, Paul Fortier and Benoit Gosselin}
    }
