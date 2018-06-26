import numpy as np
np.set_printoptions(threshold=np.nan)
import random
import h5py as h5
import tensorflow as tf
from parse_data import *

######## CNN MODEL ########
x = tf.placeholder(tf.float32, shape = [None, 30000])
y = tf.placeholder(tf.float32, shape = [None, 3])

def model(x):
        #Reshape input
        inp = tf.reshape(x, [-1, 30000, 1, 1])

        #Convolutional layer 1,2
        w_conv1 = weight('w_conv1', [80, 1, 1, 64])
        b_conv1 = bias('b_conv1', [64])
        conv1 = tf.nn.elu(conv(inp, w_conv1) + b_conv1)

	w_conv2 = weight('w_conv2', [20, 1, 64, 64])
	b_conv2 = bias('b_conv2', [64])
	conv2 = tf.nn.elu(conv(conv1, w_conv2) + b_conv2)

        #Convolutional layer 3,4
        w_conv3 = weight('w_conv3', [10, 1, 64, 64])
        b_conv3 = bias('b_conv3', [64])
        conv3 = tf.nn.elu(conv(conv2, w_conv3) + b_conv3)

        w_conv4 = weight('w_conv4', [10, 1, 64, 64])
        b_conv4 = bias('b_conv4', [64])
        conv4 = conv(conv2, w_conv3) + b_conv3

        #Residual Layer 1
        conv4 = tf.nn.elu(conv4 + conv2)

        #Max pool 1
        conv4 = maxPool(conv4, 5)
	print(conv4.shape)

        #Convolutional layer 5,6
        w_conv5 = weight('w_conv5', [5, 1, 64, 128])
        b_conv5 = bias('b_conv5', [128])
        conv5 = tf.nn.elu(conv(conv4, w_conv5) + b_conv5)

        w_conv6 = weight('w_conv6', [5, 1, 128, 128])
        b_conv6 = bias('b_conv6', [128])
        conv6 = tf.nn.elu(conv(conv5, w_conv6) + b_conv6)

        #Convolutional layer 7, 8
        w_conv7 = weight('w_conv7', [5, 1, 128, 128])
        b_conv7 = bias('b_conv7', [128])
        conv7 = tf.nn.elu(conv(conv6, w_conv7) + b_conv7)

        w_conv8 = weight('w_conv8', [5, 1, 128, 128])
        b_conv8 = bias('b_conv8', [128])
        conv8 = (conv(conv7, w_conv8) + b_conv8)

        #Residual Layer 2
        conv8 = tf.nn.elu(conv8 + conv6)

        #Max pool 2
        conv8 = maxPool(conv4, 5)
	print(conv8.shape)

        #Convolutional layer 9
        w_conv9 = weight('w_conv9', [5, 1, 128, 256])
        b_conv9 = bias('b_conv9', [256])
        conv9 = tf.nn.elu(conv(conv8, w_conv9) + b_conv9)

        #Max pool 3
        conv9 = maxPool(conv9, 5)
	print(conv9.shape)

	#Convolutional layer 10
	w_conv10 = weight('w_conv10', [5, 1, 256, 256])
	b_conv10 = bias('b_conv10', [256])
	conv10 = tf.nn.elu(conv(conv9, w_conv10) + b_conv10)

	#Max pool 4
	conv10 = maxPool(conv10, 5)
	print(conv10.shape)

	#Convolution layer 11
	w_conv11 = weight('w_conv11', [5, 1, 256, 256])
	b_conv11 = bias('b_conv11', [256])
	conv11 = tf.nn.elu(conv(conv10, w_conv11) + b_conv11)

	#Max pool 5
	conv11 = maxPool(conv11, 4)
	print(conv11.shape)

        #Flatten
        flat = tf.reshape(conv7, [-1, 12 * 256])

	#Fully connected layer
	w_fc = weight('w_fc', [12 * 256, 20])
	b_fc = bias('b_fc', [20])
	fc = tf.nn.elu(tf.matmul(flat, w_fc) + b_fc)

        #Output layer
        w_fc1 = weight('w_fc1', [20, 3])
        b_fc1 = bias('b_fc1', [3])
        prediction = tf.nn.sigmoid(tf.matmul(fc, w_fc1) + b_fc1)

        #Training neural network
        epochs = 100
        epsilon = 10
        cost = (tf.losses.mean_squared_error(prediction, y))

        #mse_q = (tf.losses.mean_squared_error(prediction[:,0], y[:,0]))
        #mse_s1z = (tf.losses.mean_squared_error(prediction[:,1], y[:,1]))
        #mse_s2z = (tf.losses.mean_squared_error(prediction[:,2], y[:,2]))
        re_q = tf.divide(tf.abs(tf.subtract(prediction[:,0], y[:,0])), y[:,0]) * 100  #Relative error
        re_s1z = tf.divide(tf.abs(tf.subtract(prediction[:,1], y[:,1])), y[:,1]) * 100  #Relative error
	re_s2z = tf.divide(tf.abs(tf.subtract(prediction[:,2], y[:,2])), y[:,2]) * 100  #Relative error

	q = tf.reduce_mean(re_q)
	s1z = tf.reduce_mean(re_s1z)
	s2z = tf.reduce_mean(re_s2z)

        optimizer = tf.train.AdamOptimizer(.0001).minimize(cost)
	batch_size = 5
        config = tf.ConfigProto(device_count = {'GPU': 0}) #Use CPU instead of GPU

        with tf.Session(config = config) as sess:
                print("Starting TensorFlow session...")
                sess.run(tf.global_variables_initializer())
                sample, label = process_data('train50.h5')
                print(sample.shape, label.shape)
                test_samples, test_labels = process_data('test.h5')
                print("Processed data!")
                graph_cost = []
                graph_epoch = []
                total_size = (sample.shape)[0]
                print("Number of samples: ", total_size)

                for epoch in range(epochs):
                        cost_ = 0
                        i = 0
			temp_sample = np.copy(sample)
			temp_label = np.copy(label)

                        np.random.seed(epoch%1000)
                        np.random.shuffle(temp_sample)
			np.random.seed(epoch%1000)
                        np.random.shuffle(temp_label)


                        while i < total_size:
				batch_sample = temp_sample[i:i+batch_size]
                                batch_label = temp_label[i:i+batch_size]

                                _, c = sess.run([optimizer, cost], feed_dict = {x: batch_sample, y: batch_label})
				i += batch_size
                                cost_ += c/(total_size/batch_size)

                        if epoch % 10 == 0:
                                graph_epoch.append(epoch)
                                graph_cost.append(c)
                                print(str(epoch + 10) + " out of " + str(epochs) + " completed. Loss: " + str(c))

                print("Optimization complete...\n")
                print("Training set predictions: ")
		print(prediction.eval({x: sample}))
                print(label)

		print("Test set predictions: ")
		print(prediction.eval({x: test_samples}))
		print(test_labels)

                print("Relative Error for Q: ",  q.eval({x: sample, y: label}))
                print("Relative Error for s1z: ", s1z.eval({x: sample, y: label}))
                print("Relative Error for s2z: ", s2z.eval({x: sample, y: label}))

                correct = (re_s2z < epsilon) #see if the difference is less than the threshold
                correct = tf.cast(correct, tf.float32)         #convert boolean tensor to float32
                accuracy = tf.reduce_mean(correct, axis=None)*100

                print('Training set accuracy: ', (accuracy.eval({x: sample, y: label}), '%'))
                print('Test set accuracy: ', (accuracy.eval({x: test_samples, y: test_labels}), '%'))


model(x)





