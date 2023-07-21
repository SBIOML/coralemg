import argparse
import time
import numpy as np
import tflite_runtime.interpreter as tflite
import platform
from sklearn import preprocessing

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]


def reduce_nb_channels(data):

    time = np.shape(data)[0]
    tmp_data = np.array(data.reshape(time,16,16))
    channels = 64
    reduced_data = np.zeros((time,channels))

    for i in range(8):
        for j in range(8):
                reduced_data[:,j+8*i] = (tmp_data[:,j*2,i*2]+tmp_data[:,1+j*2,i*2]+tmp_data[:,j*2,1+i*2]+tmp_data[:,1+j*2,1+i*2])/4

    return reduced_data

def input_details(interpreter, key):
  """Returns input details by specified key."""
  return interpreter.get_input_details()[0][key]

def input_size(interpreter):
  """Returns input image size as (width, height) tuple."""
  _, height, width, _ = input_details(interpreter, 'shape')
  return width, height

def input_tensor(interpreter):
  """Returns input tensor view as numpy array of shape (height, width, 1)."""
  tensor_index = input_details(interpreter, 'index')
  return interpreter.tensor(tensor_index)()[0]

def set_input(interpreter, data):
  """Copies data to input tensor."""
  input_tensor(interpreter)[:, :] = data

def output_tensor(interpreter, dequantize=True):
  """Returns output tensor of classification model.
  Integer output tensor is dequantized by default.
  Args:
    interpreter: tflite.Interpreter;
    dequantize: bool; whether to dequantize integer output tensor.
  Returns:
    Output tensor as numpy array.
  """
  output_details = interpreter.get_output_details()[0]
  output_data = np.squeeze(interpreter.tensor(output_details['index'])())

  if dequantize and np.issubdtype(output_details['dtype'], np.integer):
    scale, zero_point = output_details['quantization']
    return scale * (output_data - zero_point)

  return output_data

def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])

def make_inference(interpreter, inference_data):
  start = time.perf_counter()
  set_input(interpreter, inference_data)
  interpreter.invoke()
  inference_time = time.perf_counter() - start
  scores = output_tensor(interpreter)
  print('%.1fms' % (inference_time * 1000))
  print("Experiment : %d"%(np.argmax(scores)))
  return np.argmax(scores)

def main():
    model_path = "model/correlation_model_v1_edgetpu.tflite"
    data_path = "tflite_data_test.npz"



    interpreter = make_interpreter(model_path)
    interpreter.allocate_tensors()
    size = input_size(interpreter)
    #random_data = np.random.rand(50,256)
    #random_data_reduced = reduce_nb_channels(random_data)
    #corr_matrix = np.corrcoef(random_data_reduced, rowvar = False)
    #corr_matrix = np.uint8(preprocessing.minmax_scale(corr_matrix)*255).reshape(64,64,1)

    with np.load(data_path) as data:
      time_data = data['data']
      labels = data['label']

    
    nb_inference = int(len(time_data)/50)

    print('----INFERENCE TIME----')
    print('Note: The first inference on Edge TPU is slow because it includes',
          'loading the model into Edge TPU memory.')
    for i in range(nb_inference):
      left = (0+i*50)
      right = (50+i*50)
      curr_data = time_data[left:right,:]
      curr_data_reduced = reduce_nb_channels(curr_data)
      corr_matrix = np.corrcoef(curr_data_reduced, rowvar = False)
      corr_matrix = np.uint8(preprocessing.minmax_scale(corr_matrix)*255).reshape(64,64,1)
      score = make_inference(interpreter, corr_matrix)
      print("True label : %d\n"%(labels[left]))

if __name__ == '__main__':
    main()
