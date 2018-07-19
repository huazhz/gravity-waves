import numpy as np
import random
import h5py as h5
import tensorflow as tf
from scipy.interpolate import UnivariateSpline
from scipy.signal import decimate

#np.set_printoptions(threshold=np.nan)

######## DATA PREPROCESSING ########
def process_data(filename):
        f = h5.File(filename)
        dset = []
        q = []
        s1z = []
        s2z = []

        for key in f.keys():
                #format and truncate the wave
                data = np.array(f[key])
                data = np.reshape(data, (1, -1))
                data = np.squeeze(data)
                data = data[-7500:]
		
		#interpolate
		#new_length = 135000
		#old_indices = np.arange(0,len(data))
		#new_indices = np.linspace(0,len(data)-1,new_length)
		#spl = UnivariateSpline(old_indices,data,k=3,s=0)
		#new_array = spl(new_indices)
		
		#sample
		#data = decimate(new_array, 9, ftype = 'iir')
		#data = data.astype(np.float)

                #subtract mean, normalize amplitude
                mean = np.mean(data)
                data = data - mean
                peak = np.amax(np.abs(data))
                data = data/peak
                dset.append(data)

                #extract each BBH parameter from the key name
                q.append(float(key[2:6]))
                s1z.append(float(key[11:15]))
                s2z.append(float(key[-8:-4]))

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
