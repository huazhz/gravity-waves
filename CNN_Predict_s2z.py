from scipy import interpolate
import numpy as np
import random
import h5py as h5
import os
#import tensorflow as tf


######## DATA PREPROCESSING ########
def generate_data(filename):
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
		dset.append(data[0:15000]) 

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
	#print(labels[:,1])

	return dset, labels

dset, labels = generate_data(train.h5)
print(dset.shape)
print(labels.shape)

######## HELPER FUNCTIONS ########
# def weight(name, shape):
# 	return tf.get_variable(name, shape=shape, initializer = tf.contrib.layers.xavier_initializer())

# def bias(name, shape):
# 	return tf.Variable(tf.random_normal([shape]), name=name)

# def conv(x, W):
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')

# def maxPool(x):
#     return tf.nn.max_pool(x, ksize=[1, 5, 1, 1], strides=[1, 5, 1, 1], padding = 'SAME')


######## CNN MODEL ########
# with tf.name_scope('Input'):
# 	X = tf.placeholder(tf.float32, shape = [None, 15000])

# with tf.name_scope('Labels'):
# 	Y = tf.placeholder(tf.float32, shape = [None, 3])

# with tf.name_scope('Reshape_input'):
# 	inp = tf.reshape(X, [-1, 15000, 1, 1])

# with tf.name_scope('Conv1'):
# 	w_conv1 = weight('w_conv1', [16, 1, 1, 64])
# 	b_conv1 = bias('b_conv1', [64])
# 	conv1 = conv(inp, w_conv1) + b_conv1
# 	conv1 = tf.nn.relu(conv1)

# with tf.name_scope('MaxPool1'):
# 	conv1 = maxPool(conv1)

# with tf.name_scope('Conv2'):
# 	w_conv2 = weight('w_conv2', [16, 1, 64, 128])
# 	b_conv2 = bias('b_conv2', [128])
# 	conv2 = conv(conv1, w_conv2) + b_conv2
# 	conv2 = tf.nn.relu(conv2)

# with tf.name_scope('MaxPool2'):
# 	conv2 = maxPool(conv2)

# with tf.name_scope('Conv3'):
# 	w_conv3 = weight('w_conv3', [16, 1, 128, 256])
# 	b_conv3 = bias('b_conv3', [256])
# 	conv3 = conv(conv2, w_conv3) + b_conv3
# 	conv3 = tf.nn.relu(conv3)

# with tf.name_scope('MaxPool3'):
# 	conv3 = maxPool(conv3)

# with tf.name_scope('Flatten_layer'):
# 	flat = tf.reshape(conv3, [-1, 120 * 256])

# with tf.name_scope('FC_layer'):
# 	w_fc1 = weight('w_fc1', [120 * 256, 1024])
# 	b_fc1 = bias('b_fc1', [1024])
# 	fc = tf.matmul(flat, w_fc1) + b_fc1
# 	fc = tf.nn.relu(fc)

# with tf.name_scope('Output'):
# 	w_fc2 = weight('w_f2', [1024, 3])
# 	b_fc1 = bias('b_fc2', [3])
# 	Y_ = tf.matmul(fc, w_fc2) + b_fc2


# #Loss function
# cost = tf.losses.mean_squared_error(Y_, Y)
# optimizer = tf.train.AdamOptimizer().minimize(cost)









