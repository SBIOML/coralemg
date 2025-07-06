import data_processing as dp
import dataset_definition as dtdef
import numpy as np
from micropython import const
import tpu_inference as infer
import sklearn.metrics as metrics
from pycoral.adapters import common

_TIME_LENGTH = const(25)
_VOTE_LENGTH = const(150)

def create_processed_data(dataset, raw_dataset_path, data_range, time_length=25):
    with np.load(raw_dataset_path) as data:
        raw_data = data['data']

    raw_data = raw_data[:,data_range,:,:]
    raw_data = dp.preprocess_data(raw_data, window_length=time_length, filtering_utility=not dataset.utility_filtered)
    X, y = dp.extract_with_labels(raw_data)
    y = np.array(y, dtype=np.uint8)
    return X, y

def model_evaluation(dataset, model_name, subject, session, compression_method, residual_bits, fine_tuned=False, on_device=False, debug=False):
    model_name = "%s_%s_%s_%s_%s_%sbits"%(dataset.name, model_name, subject, session, compression_method, residual_bits)

    test_session = "2" if session == "1" else "1"

    # Create data
    dataset_path = "/home/mendel/dataset/%s/%s_%s_raw.npz"%(dataset.name, subject, test_session)

    model_accuracy = []
    model_accuracy_maj = []
    confusion_list = []
    confusion_list_maj = []

    line_dim = dataset.sensors_dim[0]
    column_dim = dataset.sensors_dim[1]

    for i in range(5):
        # Create a range of 2 values to select fine tuning data based on i [0,1]; [2,3] ...
        fine_tuning_range = range(i*2, i*2+2)
        if fine_tuned:
            if on_device:
                running_model_name = model_name + "_ondevice_tuned_%s_%s"%(fine_tuning_range[0], fine_tuning_range[-1])
            else:
                running_model_name = model_name + "_tuned_%s_%s"%(fine_tuning_range[0], fine_tuning_range[-1])
        else:
            running_model_name = model_name
        model_path = "/home/mendel/model/%s_edgetpu.tflite"%(running_model_name)
        #get all the rest of the range for testing from  0 to 9
        testing_range = list(set(range(10)) - set(fine_tuning_range))
        X_test, y_test = create_processed_data(dataset, dataset_path, testing_range)
        X_test = dp.compress_data(X_test, method=compression_method, residual_bits=residual_bits)

        # Get majority vote for y in the time window
        nb_votes = int(np.floor(_VOTE_LENGTH/_TIME_LENGTH))
        votes_arr = np.zeros(nb_votes, dtype=np.uint8)

        y_true = y_test
        y_true_maj = np.array([np.argmax(np.bincount(y_true[i:i+nb_votes])) for i in range(0, len(y_true), nb_votes)])
        y_pred = []
        y_pred_maj = []
        interpreter = infer.make_interpreter(model_path)
        #TODO check if correct
        width, height = common.input_size(interpreter)
        interpreter.allocate_tensors()
        curr_vote = 0
        for sample_nb, x_sample in enumerate(X_test):
            reshaped_data = x_sample.reshape(height,width,1)
            vote = infer.make_inference(interpreter, reshaped_data, False)
            y_pred.append(vote)
            votes_arr[curr_vote] = vote
            if (curr_vote+1 == nb_votes) or (sample_nb == len(X_test)-1):
                majority_vote = np.argmax(np.bincount(votes_arr))
                y_pred_maj.append(majority_vote)
            curr_vote = (curr_vote+1)%nb_votes

        if debug:
            print("Number of output labels:", len(y_pred))
            print("Number of majority vote labels:", len(y_pred_maj))

        y_pred = np.array(y_pred, dtype=np.uint8)
        y_pred_maj = np.array(y_pred_maj, dtype=np.uint8)

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
        if on_device:
            filename = "/home/mendel/results/%s_evaluation_ondevice.npz"%(model_name)
        else:
            filename = "/home/mendel/results/%s_evaluation_finetuned.npz"%(model_name)
    else:
        filename = "/home/mendel/results/%s_evaluation.npz"%(model_name)
    np.savez(filename, accuracy=np.array(model_accuracy), accuracy_majority_vote=np.array(model_accuracy_maj), 
             confusion_matrix=np.array(confusion_list), confusion_matrix_maj=np.array(confusion_list_maj))
    
if __name__ == '__main__':
    #dataset = dtdef.EmagerDataset()
    dataset = dtdef.CapgmyoDataset()
    compression_methods = ["minmax","msb", "smart","root"]
    subject = "01"
    model_name = "cnn"
    sessions = ["1","2"]
    residual_bits = [1,2,3,4,5,6,7,8]
    for session in sessions:
        for compression_method in compression_methods:
            for residual_bit in residual_bits:
                model_evaluation(dataset, model_name, subject, session, compression_method, residual_bit, fine_tuned=True, on_device=True, debug=True)
