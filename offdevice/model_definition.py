import tensorflow as tf


AVG_SHAPE = (4, 16, 1)

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

def coralemg_net(nb_class):
    activation = 'relu6'
    hidden_layer = 64
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters = hidden_layer, kernel_size = 3, strides = 1, padding = 'same', activation=activation, input_shape=AVG_SHAPE),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(hidden_layer, kernel_size = 3, strides = 1, padding = 'same', activation=activation),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(hidden_layer, kernel_size = 5, strides = 1, padding = 'same', activation=activation),
        tf.keras.layers.BatchNormalization(),
		#tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation=activation),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(nb_class, activation='softmax')
    ])

    return model