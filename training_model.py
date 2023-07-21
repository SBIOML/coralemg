import tensorflow as tf
assert float(tf.__version__[:3]) >= 2.3

import os
import numpy as np
import matplotlib.pyplot as plt

BATCH_SIZE = 64

CORR_SHAPE = (64, 64, 1)
AVG_SHAPE = (4, 16, 1)

dataset_path_avg = '/home/etienne/Documents/maitrise/recherche/CORALEMG/dataset/average/ftensorflow/chan_64/window_25/all_subject.npz'
dataset_path_corr = '/home/etienne/Documents/maitrise/recherche/CORALEMG/dataset/correlation/ftensorflow/chan_64/window_50/all_subject.npz'


def train_model(dataset_path, nb_class, model_name = "average_model"):
    gpu_devices = tf.config.experimental.list_physical_devices("GPU") 
    for device in gpu_devices: 
        tf.config.experimental.set_memory_growth(device, True)


    train_dataset, test_dataset = create_dataset(dataset_path)
    

    optimizer = tf.keras.optimizers.Adam(0.0005)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=2, min_lr=0.00001)

    if "correlation" in model_name:
        model = create_correlation_model(nb_class)
    elif "average" in model_name:
        model = create_average_model(nb_class)

    model.compile(optimizer=optimizer, 
                loss='sparse_categorical_crossentropy', 
                metrics=['accuracy'])


    model.summary()
    print('Number of trainable weights = {}'.format(len(model.trainable_weights)))
    print("Number of layers in the model: ", len(model.layers))

    history = model.fit(train_dataset,
                        epochs=10,
                        validation_data=test_dataset, callbacks=[reduce_lr])

    generate_history_graph(history)

    saved_keras_model = 'model/%s.h5'%(model_name)
    model.save(saved_keras_model)

def create_correlation_model(nb_class):
    activation = 'relu'
    hidden_layer_1 = 128
    hidden_layer_2 = 64
    hidden_layer_3 = 64
    hidden_layer_4 = 32

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters = hidden_layer_1, kernel_size = 3, strides = 1, padding = 'same', activation=activation, input_shape=CORR_SHAPE),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(hidden_layer_2, kernel_size = 3, strides = 1, padding = 'same', activation=activation),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(hidden_layer_3, kernel_size = 3, strides = 1, padding = 'same', activation=activation),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(hidden_layer_4, kernel_size = (3, 3), strides = (1, 1), padding = 'same', activation=activation),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation=activation),
        tf.keras.layers.Dense(nb_class, activation='softmax')
    ])

    return model

def create_average_model(nb_class):
    activation = 'relu'
    hidden_layer_1 = 64
    hidden_layer_2 = 64
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters = hidden_layer_1, kernel_size = 3, strides = 1, padding = 'same', activation=activation, input_shape=AVG_SHAPE),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(hidden_layer_2, kernel_size = 3, strides = 1, padding = 'same', activation=activation),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=activation),
        tf.keras.layers.Dense(nb_class, activation='softmax')
    ])

    return model

def create_dataset(dataset_path):
    with np.load(dataset_path) as data:
        train_examples = data['train_data']
        train_labels = data['train_label']
        test_examples = data['test_data']
        test_labels = data['test_label']

    data_dim = np.shape(train_examples)

    SHUFFLE_BUFFER_SIZE = len(train_labels)
    train_examples = train_examples.astype('float32').reshape(-1,data_dim[1],data_dim[2],1)
    train_examples /= 255.0
    test_examples = test_examples.astype('float16').reshape(-1,data_dim[1],data_dim[2],1)
    test_examples /= 255.0

    #test_examples = test_examples[0:1000,:,:,:]
    #test_labels = test_labels[0:1000]

    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))


    train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

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

if __name__ == "__main__":
    train_model(dataset_path_avg, 6)