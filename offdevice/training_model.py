import tensorflow as tf

import os
import numpy as np
import matplotlib.pyplot as plt
import model_definition as md
import data_processing as dp
import dataset_definition as dtdef


BATCH_SIZE = 64

def scheduler(epoch, lr):
  if epoch < 5:
    return lr
  else:
    return lr * tf.math.exp(-0.5)

def train_model(dataset, model_name, dataset_path, subject="00", session="1", compression_mode="minmax", bit=8):
    '''
    Train the model

    @param dataset the class that represents the dataset
    @param model_name the name of the model
    @param dataset_path the path to the dataset
    @param subject the subject to use, must be 00, 01, ...
    @param session the session to use, must be 1, 2
    @param compression_mode compression method used
    '''
    gpu_devices = tf.config.experimental.list_physical_devices("GPU") 
    for device in gpu_devices: 
        tf.config.experimental.set_memory_growth(device, True)


    train_dataset, test_dataset = create_training_dataset(dataset, dataset_path, subject, session, compression_mode, bit)
    

    optimizer = tf.keras.optimizers.AdamW(0.00025)
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

    model_shape = (dataset.sensors_dim[0], dataset.sensors_dim[1], 1)

    model = md.coralemg_net(dataset.nb_class, model_shape)

    model.compile(optimizer=optimizer, 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])


    model.summary()
    print('Number of trainable weights = {}'.format(len(model.trainable_weights)))
    print("Number of layers in the model: ", len(model.layers))

    history = model.fit(train_dataset,
                        epochs=10,
                        validation_data=test_dataset, callbacks=[reduce_lr])

    #generate_history_graph(history)

    model_name = "%s_%s_%s_%s_%s_%sbits"%(dataset.name, model_name, subject, session, compression_mode, bit)

    saved_keras_model = 'offdevice/model/%s.h5'%(model_name)
    model.save(saved_keras_model)

def fine_tune_model(dataset, dataset_path, folder_model_path, tuned_name):
    '''
    Fine tune the model

    @param dataset the class that represents the dataset
    @param dataset_path the path to the dataset
    @param folder_model_path the path to the folder containing the model
    @param tuned_name the name of the model to fine tune
    '''

    gpu_devices = tf.config.experimental.list_physical_devices("GPU") 
    for device in gpu_devices: 
        tf.config.experimental.set_memory_growth(device, True)
    

    full_model_path = '%s/%s.h5'%(folder_model_path, tuned_name)
    print(full_model_path)

    splitted_name = tuned_name.split("_")
    subject = splitted_name[2]
    session = splitted_name[3]
    compressed_method = splitted_name[4]
    nb_bits = int(splitted_name[5].split("bits")[0])

    fine_tuning_session = "2" if session == "1" else "1"

    raw_dataset_path = '%s/%s_%s_raw.npz'%(dataset_path, subject, fine_tuning_session)
    
    with np.load(raw_dataset_path) as data:
        raw_data = data['data']

    time_length = 25
    window_length = int(dataset.sampling_rate*time_length/1000)
    
    averages_data = dp.preprocess_data(raw_data, window_length=window_length, filtering_utility=not dataset.utility_filtered)

    for i in range(5):
        # load the model
        model = tf.keras.models.load_model(full_model_path)
        model.trainable = True
        print("Number of layers in the base model: ", len(model.layers))
        fine_tune_at = 7
        for layer in model.layers[:fine_tune_at]:
            layer.trainable =  False

        optimizer = tf.keras.optimizers.AdamW(0.0002)
        reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

        model.compile(optimizer=optimizer, 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy'])

        model.summary()
        # Create a range of 2 values to select fine tuning data based on i [0,1]; [2,3] ...
        fine_tuning_range = range(i*2, i*2+2)
        #get all the rest of the range for testing from  0 to 9
        testing_range = list(set(range(10)) - set(fine_tuning_range))

        fine_tuning_data = averages_data[:,fine_tuning_range,:,:]
        testing_data = averages_data[:,testing_range,:,:]
        train_dataset, test_dataset = create_tuning_dataset(dataset, fine_tuning_data, testing_data, compressed_method, nb_bits, BATCH_SIZE)

        history = model.fit(train_dataset,
                            epochs=10,
                            validation_data=test_dataset)
        
        tuned_model_name = tuned_name + "_tuned_%s_%s"%(fine_tuning_range[0], fine_tuning_range[-1])
        saved_keras_model = 'offdevice/model/tuned/%s.h5'%(tuned_model_name)
        model.save(saved_keras_model)

        #generate_history_graph(history)


def create_training_dataset(dataset, dataset_path, subject="00", session="1", compression_mode="minmax", bit=8):

    test_session = "2" if session == "1" else "1"

    train_dataset_path = '%s/%s_%s_%s_%sbits.npz'%(dataset_path, subject, session, compression_mode, bit)
    test_dataset_path = '%s/%s_%s_%s_%sbits.npz'%(dataset_path, subject, test_session, compression_mode, bit)

    with np.load(train_dataset_path) as data:
        X_train = data['data']
        y_train = data['label']

    with np.load(test_dataset_path) as data:
        nb_of_data = np.shape(data['label'])[0]
        lw_bound = int(2*nb_of_data/5)
        hp_bound = int(3*nb_of_data/5)
        X_test = data['data'][lw_bound:hp_bound,:]
        y_test = data['label'][lw_bound:hp_bound]

    SHUFFLE_BUFFER_SIZE = len(y_train)
    X_train = X_train.astype('float32').reshape(-1,dataset.sensors_dim[0],dataset.sensors_dim[1],1)
    X_test = X_test.astype('float32').reshape(-1,dataset.sensors_dim[0],dataset.sensors_dim[1],1)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(BATCH_SIZE)

    return train_dataset, test_dataset

def create_tuning_dataset(dataset, finetuning_data, testing_data, compression_method, nb_bits, batch_size=64):
    X_train, y_train = dp.extract_with_labels(finetuning_data)
    X_test, y_test = dp.extract_with_labels(testing_data)

    X_train = dp.compress_data(X_train, method=compression_method, residual_bits=nb_bits)
    X_train = X_train.astype('float32').reshape(-1,dataset.sensors_dim[0],dataset.sensors_dim[1],1)

    X_test = dp.compress_data(X_test, method=compression_method, residual_bits=nb_bits)
    X_test = X_test.astype('float32').reshape(-1,dataset.sensors_dim[0],dataset.sensors_dim[1],1)

    y_train = np.array(y_train, dtype=np.uint8)
    y_test = np.array(y_test, dtype=np.uint8)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(batch_size)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, test_dataset

def generate_history_graph(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

def train_all_subjects(dataset, model_name, dataset_path, subjects, compression_methods, bits):
    for subject in subjects:
        for j in range(2):
            for compression in compression_methods:
                for bit in bits:
                    data_path_compressed = dataset_path+"%s"%(compression)
                    session = str(j+1)
                    train_model(dataset, model_name, data_path_compressed, subject, session, compression, bit)

def finetune_all_subjects(dataset, model_name, dataset_path, subjects, folder_model_path, compression_methods, bits):
    dataset_name = dataset.name
    for subject in subjects:
        for j in range(2):
            for compression in compression_methods:
                for bit in bits:
                    session = str(j+1)
                    tuned_name = "%s_%s_%s_%s_%s_%sbits"%(dataset_name, model_name, subject, session, compression, bit)
                    fine_tune_model(dataset, dataset_path, folder_model_path, tuned_name)

if __name__ == "__main__":

    dataset = dtdef.EmagerDataset()
    dataset_name = dataset.name
    subjects = ["00","01","02","03","04","05","06","07","08","09","10", "11"]
    model_name = "cnn"
    train_dataset_path= 'dataset/train/%s/'%(dataset_name)
    compression_methods = ["baseline"]
    #compression_methods = ["minmax", "msb", "smart", "root"]
    bits = [1,2,3,4,5,6,7,8]
    train_all_subjects(dataset, model_name, train_dataset_path, subjects, compression_methods, bits)


    raw_dataset_path = 'dataset/raw/%s'%(dataset_name)
    folder_model_path = 'offdevice/model'
    finetune_all_subjects(dataset, model_name, raw_dataset_path, subjects, folder_model_path, compression_methods, bits)

    dataset = dtdef.CapgmyoDataset()
    dataset_name = dataset.name
    subjects = ["01","02","03","04","05","06","07","08","09","10"]
    model_name = "cnn"
    train_dataset_path= 'dataset/train/%s/'%(dataset_name)
    compression_methods = ["baseline"]
    #compression_methods = ["minmax", "msb", "smart", "root"]
    bits = [1,2,3,4,5,6,7,8]
    train_all_subjects(dataset, model_name, train_dataset_path, subjects, compression_methods, bits)


    raw_dataset_path = 'dataset/raw/%s'%(dataset_name)
    folder_model_path = 'offdevice/model'
    finetune_all_subjects(dataset, model_name, raw_dataset_path, subjects, folder_model_path, compression_methods, bits)
