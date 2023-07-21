import numpy as np
import tensorflow as tf
from sklearn import preprocessing
import time 

random_data = np.random.rand(50,256)

def reduce_nb_channels(data):

    time = np.shape(data)[0]
    tmp_data = np.array(data.reshape(time,16,16))
    channels = 64
    reduced_data = np.zeros((time,channels))

    for i in range(8):
        for j in range(8):
                reduced_data[:,j+8*i] = (tmp_data[:,j*2,i*2]+tmp_data[:,1+j*2,i*2]+tmp_data[:,j*2,1+i*2]+tmp_data[:,1+j*2,1+i*2])/4

    return reduced_data


start_time = time.time()
random_data_reduced = reduce_nb_channels(random_data)
corr_matrix = np.corrcoef(random_data_reduced, rowvar = False)
corr_matrix = np.uint8(preprocessing.minmax_scale(corr_matrix)*255)
print(time.time()-start_time)

path = "/home/etienne/Downloads/mnist.npz"
with np.load(path) as data:
  train_examples = data['x_train']
  train_labels = data['y_train']
  test_examples = data['x_test']
  test_labels = data['y_test']