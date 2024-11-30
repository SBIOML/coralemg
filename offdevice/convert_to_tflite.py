import tensorflow as tf
from tensorflow.keras.models import Model
import numpy as np
import dataset_definition as dtdef


class RepresentativeDataGenerator():
    def __init__(self, dataset, bit):
        self.hdim = dataset.sensors_dim[0]
        self.vdim = dataset.sensors_dim[1]
        self.bit = bit
    def yield_representative_data(self):
        data_examples = np.random.randint(0, 256, size=(100, self.hdim, self.vdim, 1))
        data_examples = data_examples.astype('float32')
        for data in data_examples:
            yield [data.reshape(1,self.hdim, self.vdim,1)]  

# def yield_representative_data():
#     '''
#     Yield the representative data sample for the tflite model

#     @param dataset_path the path to the dataset
#     @param subject the subject to use, must be 000, 001, ...
#     @param session the session to use, must be 001, 002
#     @param compression_mode compression method used
#     '''
#     data_examples = np.random.randint(0, 256, size=(100, 4, 16, 1))

#     data_examples = data_examples.astype('float32')
#     for data in data_examples:
#         yield [data.reshape(1,4,16,1)]
        
def convert_model_to_tflite(folder_path, model_name, model_type, dataset, bit, tflite_model_name):
    '''
    Convert a keras model to a tflite model

    @param folder_path the path to the folder containing the model
    @param model_name the name of the model to convert
    @param model_type the type of model to convert (normal, tuned, extractor)
    @param tflite_model_name the name of the tflite model to save

    @return the converted tflite model
    '''

    data_generator = RepresentativeDataGenerator(dataset, bit)
    print("Converting")
    model_path = '%s/%s.h5'%(folder_path, model_name)
    saved_keras_model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(saved_keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    converter.representative_dataset = data_generator.yield_representative_data
    tflite_model = converter.convert()

    with open('offdevice/model/tflite/%s/%s.tflite'%(model_type, tflite_model_name), 'wb') as f:
        f.write(tflite_model)

def convert_extractor_to_tflite(folder_path, model_name, model_type, dataset, bit, tflite_model_name):
    '''
    Convert a keras model to a tflite model

    @param folder_path the path to the folder containing the model
    @param model_name the name of the model to convert
    @param tflite_model_name the name of the tflite model to save

    @return the converted tflite model
    '''

    data_generator = RepresentativeDataGenerator(dataset, bit)

    model_path = '%s/%s.h5'%(folder_path, model_name)
    saved_keras_model = tf.keras.models.load_model(model_path)
    extractor_model = Model(saved_keras_model.input, saved_keras_model.layers[-2].output)


    converter = tf.lite.TFLiteConverter.from_keras_model(extractor_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    converter.representative_dataset = data_generator.yield_representative_data
    tflite_model = converter.convert()

    with open('offdevice/model/tflite/extractor/%s_ondevice.tflite'%(tflite_model_name), 'wb') as f:
        f.write(tflite_model)


if __name__ == '__main__':
    # convert the model

    dataset = dtdef.CapgmyoDataset()
    bits = [1,2,3,4,5,6,7,8]

    compression_methods = ["minmax", "msb", "smart", "root"]

    folder_path = "offdevice/model"
    model_tuning = "normal"
    model_type = "cnn"
    for sub in range(1,11):
        for sess in range(2):
            for compression_mode in compression_methods:
                for bit in bits:
                    subject = "0" + str(sub) if sub < 10 else str(sub)
                    session = str(sess+1)
                    model_name = "%s_%s_%s_%s_%s_%sbits"%(dataset.name, model_type, subject, session, compression_mode, bit)
                    tflite_model_name = model_name
                    convert_model_to_tflite(folder_path, model_name, model_tuning, dataset, bit, tflite_model_name)

    # folder_path = "offdevice/model/tuned"
    # model_tuning = "tuned"
    # model_type = "cnn"
    # for sub in range(12):
    #     subject = "0" + str(sub) if sub < 10 else str(sub)
    #     for sess in range(2):
    #         session = str(sess+1)
    #         for tuning in range(5):
    #             fine_tuning_range = range(tuning*2, tuning*2+2)
    #             for compression_mode in compression_methods:
    #                 dataset_path = '/dataset/train/%s/'%(compression_mode)
    #                 model_name = "%s_%s_%s_%s_%s_%sbits_tuned_%s_%s"%(dataset.name, model_type, subject, session, compression_mode, bit, fine_tuning_range[0], fine_tuning_range[-1])
    #                 tflite_model_name = model_name
    #                 convert_model_to_tflite(folder_path, model_name, model_tuning, tflite_model_name)


    # folder_path = "offdevice/model"
    # model_tuning = "normal"
    # for sub in range(12):
    #     for sess in range(2):
    #         for compression_mode in compression_methods:
    #             dataset_path = 'dataset/train/%s/'%(compression_mode)
    #             subject = "0" + str(sub) if sub < 10 else str(sub)
    #             session = str(sess+1)
    #             model_name = "%s_%s_%s_%s_%s_%sbits"%(dataset.name, model_type, subject, session, compression_mode, bit)
    #             tflite_model_name = model_name
    #             convert_extractor_to_tflite(folder_path, model_name, model_tuning, tflite_model_name)
