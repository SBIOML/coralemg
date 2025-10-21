import os
import numpy as np
import tensorflow as tf
import data_processing as dp
import sklearn.metrics as metrics
import dataset_definition as dtdef


def create_processed_data(dataset, raw_dataset_path, data_range, time_length=25):
    with np.load(raw_dataset_path) as data:
        raw_data = data['data']

    raw_data = raw_data[:,data_range,:,:]
    raw_data = dp.preprocess_data(raw_data, window_length=time_length, filtering_utility=not dataset.utility_filtered)
    X, y = dp.extract_with_labels(raw_data)
    y = np.array(y, dtype=np.uint8)
    return X, y

def evaluate_raw_model(dataset, model_path, dataset_path, model_name):
    BATCH_SIZE = 64

    model_path = '%s/%s.h5'%(model_path, model_name)
    model = tf.keras.models.load_model(model_path)

    with np.load(dataset_path) as data:
        test_examples = data['data']
        test_labels = data['label']
    
    line_dim = dataset.sensors_dim[0]
    column_dim = dataset.sensors_dim[1]
    test_examples = test_examples.astype('float32').reshape(-1,line_dim,column_dim,1)

    test_dataset = tf.data.Dataset.from_tensor_slices(test_examples)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    logits = model.predict(test_dataset)

    prediction = np.argmax(logits, axis=1)
    truth = test_labels

    keras_accuracy = tf.keras.metrics.Accuracy()
    keras_accuracy(prediction, truth)

    print("Raw model accuracy: {:.3%}".format(keras_accuracy.result()))

def evaluate_tflite_model(dataset, model_path, dataset_path, tflite_model_name):
    with np.load(dataset_path) as data:
        test_examples = data['data']
        test_labels = data['label']

    line_dim = dataset.sensors_dim[0]
    column_dim = dataset.sensors_dim[1]
    test_examples = test_examples.astype(np.uint8).reshape(-1,line_dim,column_dim,1)
    data = test_examples[:,:,:,:]
    
    model_path = '%s/%s.tflite'%(model_path, tflite_model_name)
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    prediction = []
    for i in range(len(data)):
        curr_data = np.expand_dims(data[i], axis=0)
        input_tensor= tf.convert_to_tensor(curr_data, np.uint8)
        interpreter.set_tensor(input_details['index'], curr_data)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details['index'])
        prediction.append(np.argmax(output))

    tflite_accuracy = tf.keras.metrics.Accuracy()
    tflite_accuracy(prediction, test_labels)
    print("Quant TF Lite accuracy: {:.3%}".format(tflite_accuracy.result()))

def test_model_performance(dataset, model_path, raw_dataset_path, result_path, model_name, subject, session, compression_method, bit, fine_tuned=False, time_length=25, vote_length=150):
    BATCH_SIZE = 64
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    dataset_name = dataset.name

    model_name = "%s_%s_%s_%s_%s_%sbits"%(dataset_name, model_name, subject, session, compression_method, bit)
    test_session = "2" if session == "1" else "1"    

    # Create data
    dataset_path = '%s/%s_%s_raw.npz'%(raw_dataset_path, subject, test_session)

    model_accuracy = []
    model_accuracy_maj = []
    confusion_list = []
    confusion_list_maj = []

    for i in range(5):
        fine_tuning_range = range(i*2, i*2+2)
        if fine_tuned:
            running_model_name = model_name + "_tuned_%s_%s"%(fine_tuning_range[0], fine_tuning_range[-1])
            current_model_path = '%s/tuned/%s.h5'%(model_path, running_model_name)
        else:
            running_model_name = model_name
            current_model_path = '%s/%s.h5'%(model_path, running_model_name)
        testing_range = list(set(range(10)) - set(fine_tuning_range))
        X_test, y_test = create_processed_data(dataset, dataset_path, testing_range)
        X_test = dp.compress_data(X_test, method=compression_method, residual_bits=bit) 

        line_dim = dataset.sensors_dim[0]
        column_dim = dataset.sensors_dim[1]
        X_test = X_test.astype('float32').reshape(-1,line_dim,column_dim,1)
        nb_votes = int(np.floor(vote_length/time_length))

        y_true = y_test
        y_true_maj = np.array([np.argmax(np.bincount(y_true[i:i+nb_votes])) for i in range(0, len(y_true), nb_votes)])

        test_dataset = tf.data.Dataset.from_tensor_slices(X_test)
        test_dataset = test_dataset.batch(BATCH_SIZE)

        model = tf.keras.models.load_model(current_model_path)
        logits = model.predict(test_dataset)

        y_pred = np.argmax(logits, axis=1)
        y_pred_maj = np.array([np.argmax(np.bincount(y_pred[i:i+nb_votes])) for i in range(0, len(y_pred), nb_votes)])

        # Evaluate model performance
        accuracy = metrics.accuracy_score(y_true, y_pred)
        accuracy_maj = metrics.accuracy_score(y_true_maj, y_pred_maj)

        confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
        confusion_matrix_maj = metrics.confusion_matrix(y_true_maj, y_pred_maj)

        print("Accuracy: %.2f%%"%(accuracy*100))
        print("Accuracy majority vote: %.2f%%"%(accuracy_maj*100))
        model_accuracy.append(accuracy)
        model_accuracy_maj.append(accuracy_maj)
        confusion_list.append(confusion_matrix)
        confusion_list_maj.append(confusion_matrix_maj)

    if fine_tuned:
        filename = "%s/%s_evaluation_finetuned.npz"%(result_path, model_name)
    else:
        filename = "%s/%s_evaluation.npz"%(result_path, model_name)
    np.savez(filename, accuracy=np.array(model_accuracy), accuracy_majority_vote=np.array(model_accuracy_maj), 
             confusion_matrix=np.array(confusion_list), confusion_matrix_maj=np.array(confusion_list_maj))
    
if __name__ == '__main__':
    # folder_path = "offdevice/model/"
    # tflite_path = "offdevice/model/tflite/normal/"
    # model_name = "emager_cnn_01_1_smart_5bits"
    # model_name_tflite = "emager_cnn_01_1_smart_5bits"
    # dataset_path = "dataset/train/emager/smart/01_1_smart_5bits.npz"

    # dataset = dtdef.EmagerDataset()

    # evaluate_raw_model(dataset, folder_path, dataset_path, model_name)
    # evaluate_tflite_model(dataset, tflite_path, dataset_path, model_name_tflite)

    dataset = dtdef.EmagerDataset()
    dataset_name = dataset.name
    model_name = "cnn"
    subjects = ["00","01","02","03","04","05","06","07","08","09","10","11"]
    sessions = ["1", "2"]
    compression_methods = ["minmax", "msb", "smart", "root", "baseline"]
    bits = [1,2,3,4,5,6,7,8]

    model_path = "offdevice/model"
    dataset_path = "dataset/raw/%s"%(dataset_name)
    result_path = "offdevice/offdevice_results"
    for subject in subjects:
        for session in sessions:
            for compression_method in compression_methods:
                for bit in bits:
                    test_model_performance(dataset, model_path, dataset_path, result_path, model_name, subject, session, compression_method, bit, fine_tuned=False, time_length=25, vote_length=150)
                    test_model_performance(dataset, model_path, dataset_path, result_path, model_name, subject, session, compression_method, bit, fine_tuned=True, time_length=25, vote_length=150)


    dataset = dtdef.CapgmyoDataset()
    dataset_name = dataset.name
    model_name = "cnn"
    subjects = ["01","02","03","04","05","06","07","08","09","10"]
    sessions = ["1", "2"]
    compression_methods = ["minmax", "msb", "smart", "root", "baseline"]
    bits = [1,2,3,4,5,6,7,8]

    model_path = "offdevice/model"
    dataset_path = "dataset/raw/%s"%(dataset_name)
    result_path = "offdevice/offdevice_results"
    for subject in subjects:
        for session in sessions:
            for compression_method in compression_methods:
                for bit in bits:
                    test_model_performance(dataset, model_path, dataset_path, result_path, model_name, subject, session, compression_method, bit, fine_tuned=False, time_length=25, vote_length=150)
                    test_model_performance(dataset, model_path, dataset_path, result_path, model_name, subject, session, compression_method, bit, fine_tuned=True, time_length=25, vote_length=150)

