import tensorflow as tf
import numpy as np
assert float(tf.__version__[:3]) >= 2.3


folder_path = "/home/etienne/Documents/maitrise/recherche/CORALEMG/model"
model_name_corr = "correlation_model"
tflite_model_name_corr = "correlation_model_v1"
dataset_path_corr = "/home/etienne/Documents/maitrise/recherche/CORALEMG/dataset/correlation/ftensorflow/chan_64/window_50/all_subject.npz"

model_name_avg = "average_model"
tflite_model_name_avg = "average_model_v1"
dataset_path_avg = "/home/etienne/Documents/maitrise/recherche/CORALEMG/dataset/average/ftensorflow/chan_64/window_25/all_subject.npz"

def yield_representative_data():
    with np.load(dataset_path_avg) as data:
        data_examples = data['train_data']

    data_dim = np.shape(data_examples)
    np.random.shuffle(data_examples)
    data_examples = data_examples[0:100,:,:]
    data_examples = data_examples.astype('float32').reshape(-1,data_dim[1],data_dim[2],1)
    data_examples /= 255.0
    for i in range(len(data_examples)):
        tmp_data = next(iter(data_examples)).reshape(1,data_dim[1],data_dim[2],1)
        yield [tmp_data]
        
def convert_model_to_tflite(folder_path, model_name, tflite_model_name):
    model_path = '%s/%s.h5'%(folder_path, model_name)
    saved_keras_model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(saved_keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    converter.representative_dataset = yield_representative_data
    tflite_model = converter.convert()


    with open('model/%s.tflite'%(tflite_model_name), 'wb') as f:
        f.write(tflite_model)

def evaluate_raw_model(folder_path,dataset_path,model_name):
    BATCH_SIZE = 64

    model_path = '%s/%s.h5'%(folder_path, model_name)
    model = tf.keras.models.load_model(model_path)

    with np.load(dataset_path) as data:
        test_examples = data['test_data']
        test_labels = data['test_label']

    data_dim = np.shape(test_examples)
    
    test_examples = test_examples.astype('float32').reshape(-1,data_dim[1],data_dim[2],1)
    test_examples /= 255.0

    test_dataset = tf.data.Dataset.from_tensor_slices(test_examples)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    logits = model.predict(test_dataset)

    prediction = np.argmax(logits, axis=1)
    truth = test_labels

    keras_accuracy = tf.keras.metrics.Accuracy()
    keras_accuracy(prediction, truth)

    print("Raw model accuracy: {:.3%}".format(keras_accuracy.result()))

def evaluate_tflite_model(folder_path, dataset_path, tflite_model_name):
    with np.load(dataset_path) as data:
        test_examples = data['test_data']
        test_labels = data['test_label']

    data_dim = np.shape(test_examples)
    
    test_examples = test_examples.reshape(-1,data_dim[1],data_dim[2],1)
    data = test_examples[:,:,:,:]
    
    model_path = '%s/%s.tflite'%(folder_path, tflite_model_name)
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

if __name__ == '__main__':
    #convert_model_to_tflite(folder_path, model_name_avg, tflite_model_name_avg)
    evaluate_raw_model(folder_path, dataset_path_avg, model_name_avg)
    evaluate_tflite_model(folder_path, dataset_path_avg, model_name_avg+"_v1")