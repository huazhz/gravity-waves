import numpy as np
import random
import h5py as h5
import tensorflow as tf
import string

#np.set_printoptions(threshold=np.nan)

######## DATA PREPROCESSING ########
def process_data(directory):

    dset = []
    q = []
    s1z = []
    s2z = []

    for filename in os.listdir(directory):
        if filename.endswith(".dat"):

            #format and truncate the wave
            data = np.loadtxt(filename)
            data = np.reshape(data, (1, -1))
            data = np.squeeze(data)
            data = data[-7500:]
    
            #subtract mean, normalize amplitude
            mean = np.mean(data)
            data = data - mean
            peak = np.amax(np.abs(data))
            data = data/peak
            dset.append(data)

            #extract each BBH parameter from the file name
            q.append(float(filename[2:6]))
            s1z.append(float(filename[11:15]))
            s2z.append(float(filename[-8:-4]))

        #convert each list to a numpy array
        dset = np.array(dset)
        q = np.array(q)
        s1z = np.array(s1z)
        s2z = np.array(s2z)

        #reshape all arrays
        q = np.reshape(q, (1, -1))
        s1z = np.reshape(s1z, (1, -1))
        s2z = np.reshape(s2z, (1, -1))

        #combine all three parameter vectors (each column is q, s1z, s2z)
        labels = np.concatenate((q, s1z, s2z), axis=0)
        labels = np.transpose(labels)

        return dset, labels

######## HELPER FUNCTIONS ########
def weight(name, shape):
        return tf.get_variable(name, shape=shape, initializer = tf.contrib.layers.xavier_initializer())

def bias(name, shape):
        #return tf.Variable(tf.random_normal(shape), name=name)
    return tf.get_variable(name, shape=shape, initializer = tf.constant_initializer(.1))

def conv(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')

def maxPool(x, n):
    return tf.nn.max_pool(x, ksize=[1, n, 1, 1], strides=[1, n, 1, 1], padding = 'SAME')
