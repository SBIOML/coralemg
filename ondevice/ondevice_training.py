#inspired from https://github.com/google-coral/pycoral/blob/master/examples/backprop_last_layer.py

import os
import sys
import time
import numpy as np
import data_processing as dp
import dataset_definition as dtdef

from pycoral.adapters import classify
from pycoral.adapters import common
from pycoral.learn.backprop.softmax_regression import SoftmaxRegression
import tpu_inference as infer

def extract_embeddings(data_array, interpreter):
    """
    Extract model embeddings from data array.

    @param data_array the data array to extract from
    @param interpreter the TFLite interpreter to use

    @return a ndarray of embeddings
    """
    nb_data = np.shape(data_array)[0]

    #TODO check if correct
    width, height = common.input_size(interpreter)
    feature_dim = classify.num_classes(interpreter)
    embeddings = np.empty((nb_data, feature_dim), dtype=np.float32)
    for idx, data in enumerate(data_array):
      infer.set_input(interpreter, data.reshape(height,width,1))
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

def fine_tune_model(dataset, model_type, subject, session, compression_method, residual_bits):
    #TODO model_type
    model_name = "%s_%s_%s_%s_%s_%sbits_ondevice_edgetpu"%(dataset.name, model_type, subject, session, compression_method, residual_bits)

    test_session = "2" if session == "1" else "1"

    # Create data
    dataset_path = "/home/mendel/dataset/%s/%s_%s_raw.npz"%(dataset.name, subject, test_session)
    model_path = "/home/mendel/model/%s.tflite"%(model_name)

    with np.load(dataset_path) as data:
        raw_data = data['data']
    processed_data = dp.preprocess_data(raw_data, filtering_utility=not dataset.utility_filtered)

    for i in range(5):
        fine_tuning_range = range(i*2, i*2+2)
        fine_tune_data = processed_data[:,fine_tuning_range,:,:]
        X, y = dp.extract_with_labels(fine_tune_data)
        X = dp.compress_data(X, compression_method, residual_bits)
        y = np.array(y, dtype=np.uint8)
        add_to_model_name = "_tuned_%s_%s"%(fine_tuning_range[0], fine_tuning_range[-1])
        train_model(model_path, X, y, add_to_model_name)


if __name__ == "__main__":
    dataset = dtdef.CapgmyoDataset()
    model_type = "cnn"
    subject = "01"
    sessions = ["1","2"]
    compressed_methods = ["minmax", "msb", "smart", "root"]
    residual_bits = [1,2,3,4,5,6,7,8]
    for compression in compressed_methods:
        for session in sessions:
            for bits in residual_bits:
                fine_tune_model(dataset, model_type, subject, session, compression, bits)
