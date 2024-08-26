import tensorflow as tf

import os
import numpy as np
import matplotlib.pyplot as plt
import model_definition as md
import data_processing as dp


BATCH_SIZE = 64
AVG_SHAPE = (4, 16, 1)
NB_CLASSES = 6

def scheduler(epoch, lr):
  if epoch < 5:
    return lr
  else:
    return lr * tf.math.exp(-0.5)

def train_model(dataset_path, subject="000", session="001", compression_mode="minmax", bit=8, nb_class=6):
    '''
    Train the model

    @param dataset_path the path to the dataset
    @param subject the subject to use, must be 000, 001, ...
    @param session the session to use, must be 001, 002
    @param compression_mode compression method used
    @param nb_class the number of class to use    
    '''
    gpu_devices = tf.config.experimental.list_physical_devices("GPU") 
    for device in gpu_devices: 
        tf.config.experimental.set_memory_growth(device, True)


    train_dataset, test_dataset = create_training_dataset(dataset_path, subject, session, compression_mode, bit)
    

    optimizer = tf.keras.optimizers.AdamW(0.00025)
    reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)


    model = md.emager_net(NB_CLASSES)

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

    model_name = "emager_%s_%s_%s_%sbits"%(subject, session, compression_mode, bit)

    saved_keras_model = 'offdevice/model/%s.h5'%(model_name)
    model.save(saved_keras_model)

def fine_tune_model(dataset_path, folder_model_path, model_name):
    '''
    Fine tune the model

    @param dataset_path the path to the dataset
    @param folder_model_path the path to the folder containing the model
    @param model_name the name of the model to fine tune
    '''

    gpu_devices = tf.config.experimental.list_physical_devices("GPU") 
    for device in gpu_devices: 
        tf.config.experimental.set_memory_growth(device, True)
    

    full_model_path = '%s/%s.h5'%(folder_model_path, model_name)
    subject = model_name.split("_")[1]
    session = model_name.split("_")[2]
    compressed_method = model_name.split("_")[3]

    fine_tuning_session = "002" if session == "001" else "001"

    raw_dataset_path = '%s/%s_%s_raw.npz'%(dataset_path, subject, fine_tuning_session)
    
    with np.load(raw_dataset_path) as data:
        raw_data = data['data']
    
    averages_data = dp.preprocess_data(raw_data)

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
        train_dataset, test_dataset = create_tuning_dataset(fine_tuning_data, testing_data, compressed_method, BATCH_SIZE)

        history = model.fit(train_dataset,
                            epochs=10,
                            validation_data=test_dataset)
        
        tuned_model_name = model_name + "_tuned_%s_%s"%(fine_tuning_range[0], fine_tuning_range[-1])
        saved_keras_model = 'offdevice/model/tuned/%s.h5'%(tuned_model_name)
        model.save(saved_keras_model)

        #generate_history_graph(history)


def create_training_dataset(dataset_path, subject="000", session="001", compression_mode="minmax", bit=8):

    test_session = "002" if session == "001" else "001"

    train_dataset_path = '%s/%s_%s_%s_%sbits.npz'%(dataset_path, subject, session, compression_mode, bit)
    test_dataset_path = '%s/%s_%s_%s_%sbits.npz'%(dataset_path, subject, test_session, compression_mode, bit)

    with np.load(train_dataset_path) as data:
        X_train = data['data']
        y_train = data['label']

    with np.load(test_dataset_path) as data:
        X_test = data['data'][24000:36000,:]
        y_test = data['label'][24000:36000]

    SHUFFLE_BUFFER_SIZE = len(y_train)
    X_train = X_train.astype('float32').reshape(-1,4,16,1)
    X_test = X_test.astype('float32').reshape(-1,4,16,1)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(BATCH_SIZE)

    return train_dataset, test_dataset

def create_tuning_dataset(finetuning_data, testing_data, compression_method, batch_size=64):
    X_train, y_train = dp.extract_with_labels(finetuning_data)
    X_test, y_test = dp.extract_with_labels(testing_data)

    X_train = dp.compress_data(X_train, method=compression_method)
    X_train = X_train.astype('float32').reshape(-1,4,16,1)

    X_test = dp.compress_data(X_test, method=compression_method)
    X_test = X_test.astype('float32').reshape(-1,4,16,1)

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

def train_all_subjects(dataset_path, compression_methods, bits):
    for i in range(13):
        for j in range(2):
            for compression in compression_methods:
                for bit in bits:
                    data_path_compressed = dataset_path+"%s"%(compression)
                    subject = "00" + str(i) if i < 10 else "0" + str(i)
                    session = "00" + str(j+1)
                    train_model(data_path_compressed, subject, session, compression, bit)

def finetune_all_subjects(dataset_path, folder_model_path, compression_methods):
    for i in range(13):
        for j in range(2):
            for compression in compression_methods:
                for bit in bits:
                    subject = "00" + str(i) if i < 10 else "0" + str(i)
                    session = "00" + str(j+1)
                    model_name = "emager_%s_%s_%s_%sbits"%(subject, session, compression, bit)
                    fine_tune_model(dataset_path, folder_model_path, model_name, bit)

if __name__ == "__main__":
    train_dataset_path= 'dataset/train/'
    #compression_methods = ["baseline", "minmax", "msb", "smart", "root"]
    compression_methods = ["minmax", "msb", "smart", "root"]
    bits = [4,5,6,7,8]
    train_all_subjects(train_dataset_path, compression_methods, bits)


    # raw_dataset_path = 'dataset/raw/'
    # folder_model_path = 'offdevice/model'
    # finetune_all_subjects(raw_dataset_path, folder_model_path, compression_methods, bits)
