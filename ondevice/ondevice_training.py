#inspired from https://github.com/google-coral/pycoral/blob/master/examples/backprop_last_layer.py

import os
import sys
import time
import numpy as np
import data_processing as dp

from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.learn.backprop.softmax_regression import SoftmaxRegression
import tpu_inference as infer

def preprocess_data(data_array, window_length=25, fs=1000, Q=30, notch_freq=60):
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
                processed_data = dp.filter_utility(
                    processed_data, fs=fs, Q=Q, notch_freq=notch_freq
                )
                processed_data = np.mean(
                    np.absolute(processed_data - np.mean(processed_data, axis=0)),
                    axis=0,
                )
                output_data[label, experiment, curr_window, :] = processed_data
    return output_data

def extract_embeddings(data_array, interpreter):
    """
    Extract model embeddings from data array.

    @param data_array the data array to extract from
    @param interpreter the TFLite interpreter to use

    @return a ndarray of embeddings
    """
    nb_data = np.shape(data_array)[0]

    input_size = common.input_size(interpreter)
    feature_dim = classify.num_classes(interpreter)
    embeddings = np.empty((nb_data, feature_dim), dtype=np.float32)
    for idx, data in enumerate(data_array):
      infer.set_input(interpreter, data.reshape(4,16,1))
      interpreter.invoke()
      embeddings[idx, :] = classify.get_scores(interpreter)
    return embeddings

def train_model(extractor_path, X_train, y_train, add_to_model_name=""):
    """
    Train a softmax regression model given data and embedding extractor

    @param extractor_path the path to the embedding extractor
    @param X_train the training data
    @param y_train the training labels
    @param X_test the testing data, facultative
    @param y_test the testing labels, facultative
    """
    t0 = time.perf_counter()
    train_and_val_dataset = {"data_train": X_train, "labels_train": y_train, "data_val": X_train[0:20,:], "labels_val": y_train[0:20]}
    # Load the embedding extractor
    interpreter = infer.make_interpreter(extractor_path)
    interpreter.allocate_tensors()
    print('Extract embeddings for training')
    train_and_val_dataset["data_train"] = extract_embeddings(train_and_val_dataset["data_train"], interpreter)

    print('Extract embeddings for data_val')
    train_and_val_dataset['data_val'] = extract_embeddings(train_and_val_dataset['data_val'], interpreter)
    t1 = time.perf_counter()
    print('Data preprocessing takes %.2f seconds' % (t1 - t0))

    # Train the model
    weight_scale = 5e-2
    reg = 0.0
    feature_dim = train_and_val_dataset["data_train"].shape[1]
    num_classes = np.max(train_and_val_dataset["labels_train"]) + 1
    model = SoftmaxRegression(feature_dim, num_classes, weight_scale, reg)

    learning_rate = 5e-3
    num_iter = 1000
    batch_size = 100
    model.train_with_sgd(train_and_val_dataset, num_iter, learning_rate, batch_size=batch_size)
    t2 = time.perf_counter()
    print('Training takes %.2f seconds' % (t2 - t1))

    # Append learned weights to input model and save as tflite format.
    
    #remove .tflite from extractor_path
    original_model_path = extractor_path.replace("_edgetpu.tflite", "")
    out_model_path = original_model_path + add_to_model_name + "_edgetpu.tflite"
    with open(out_model_path, 'wb') as f:
        f.write(model.serialize_model(extractor_path))
    print('Model %s saved.' % out_model_path)

def fine_tune_model(subject, session, compression_method):
    model_name = "emager_%s_%s_%s_ondevice_edgetpu"%(subject, session, compression_method)

    test_session = "002" if session == "001" else "001"

    # Create data
    dataset_path = "/home/mendel/dataset/%s_%s_raw.npz"%(subject, test_session)
    model_path = "/home/mendel/model/%s.tflite"%(model_name)

    with np.load(dataset_path) as data:
        raw_data = data['data']
    processed_data = preprocess_data(raw_data)

    for i in range(5):
        fine_tuning_range = range(i*2, i*2+2)
        fine_tune_data = processed_data[:,fine_tuning_range,:,:]
        X, y = dp.extract_with_labels(fine_tune_data)
        X = dp.compress_data(X, method=compression_method)
        y = np.array(y, dtype=np.uint8)
        add_to_model_name = "_tuned_%s_%s"%(fine_tuning_range[0], fine_tuning_range[-1])
        train_model(model_path, X, y, add_to_model_name)


if __name__ == "__main__":
    subject = "012"
    sessions = ["001","002"]
    compressed_methods = ["minmax", "msb", "smart", "root"]
    for compression in compressed_methods:
        for session in sessions:
            fine_tune_model(subject, session, compression)
