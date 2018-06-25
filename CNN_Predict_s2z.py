import numpy as np
import random
import h5py as h5
import tensorflow as tf

np.set_printoptions(threshold=np.nan)

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
                data = data[0:15000]

                # subtract mean, normalize amplitude
                mean = np.mean(data)
                data = data - mean
                peak = np.maximum(np.abs(data))
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


######## CNN MODEL ########
x = tf.placeholder(tf.float32, shape = [None, 15000])
y = tf.placeholder(tf.float32, shape = [None, 3])

def model(x):
        #Reshape input
        inp = tf.reshape(x, [-1, 15000, 1, 1])

        #Convolutional layer 1,2
        w_conv1 = weight('w_conv1', [80, 1, 1, 64])
        b_conv1 = bias('b_conv1', [64])
        conv1 = tf.nn.elu(conv(inp, w_conv1) + b_conv1)

	w_conv2 = weight('w_conv2', [20, 1, 64, 64])
	b_conv2 = bias('b_conv2', [64])
	conv2 = tf.nn.elu(conv(conv1, w_conv2) + b_conv2)

        #Max pool 1
        conv2 = maxPool(conv2, 5)
	print(conv2.shape)

        #Convolutional layer 3,4
        w_conv3 = weight('w_conv3', [5, 1, 64, 128])
        b_conv3 = bias('b_conv3', [128])
        conv3 = tf.nn.elu(conv(conv2, w_conv3) + b_conv3)

        w_conv4 = weight('w_conv4', [5, 1, 128, 128])
        b_conv4 = bias('b_conv4', [128])
        conv4 = tf.nn.elu(conv(conv3, w_conv4) + b_conv4)

        #Max pool 2
        conv4 = maxPool(conv4, 5)
	print(conv4.shape)

        #Convolutional layer 5
        w_conv5 = weight('w_conv5', [5, 1, 128, 256])
        b_conv5 = bias('b_conv5', [256])
        conv5 = tf.nn.elu(conv(conv4, w_conv5) + b_conv5)

        #Max pool 3
        conv5 = maxPool(conv5, 5)
	print(conv5.shape)

	#Convolutional layer 6
	w_conv6 = weight('w_conv6', [5, 1, 256, 256])
	b_conv6 = bias('b_conv6', [256])
	conv6 = tf.nn.elu(conv(conv5, w_conv6) + b_conv6)

	#Max pool 4
	conv6 = maxPool(conv6, 5)
	print(conv6.shape)

	#Convolution layer 7
	w_conv7 = weight('w_conv7', [5, 1, 256, 256])
	b_conv7 = bias('b_conv7', [256])
	conv7 = tf.nn.elu(conv(conv6, w_conv7) + b_conv7)

	#Max pool 5
	conv7 = maxPool(conv7, 4)
	print(conv7.shape)

        #Flatten
        flat = tf.reshape(conv7, [-1, 6 * 256])

	#Fully connected layer
	w_fc = weight('w_fc', [6 * 256, 20])
	b_fc = bias('b_fc', [20])
	fc = tf.nn.elu(tf.matmul(flat, w_fc) + b_fc)

        #Output layer
        w_fc1 = weight('w_fc1', [20, 3])
        b_fc1 = bias('b_fc1', [3])
        prediction = tf.nn.sigmoid(tf.matmul(fc, w_fc1) + b_fc1)

        #Training neural network
        epochs = 5
        epsilon = .0025
        cost = (tf.losses.mean_squared_error(prediction, y))
	#cost = tf.reduce_mean(-.5 * ((1 - y)*tf.log(tf.abs(1 - tf.nn.tanh(prediction))) + (1 + y)*tf.log(tf.abs(1 + tf.nn.tanh(prediction)))))
	#cost = tf.losses.huber_loss(y, prediction, delta = epsilon)


        mse_q = (tf.losses.mean_squared_error(prediction[:,0], y[:,0]))
        mse_s1z = (tf.losses.mean_squared_error(prediction[:,1], y[:,1]))
        mse_s2z = (tf.losses.mean_squared_error(prediction[:,2], y[:,2]))
        optimizer = tf.train.AdamOptimizer(.0001).minimize(mse_s2z)

	batch_size = 10
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
		print(sample[0])
		print(sample[1])
                total_size = (sample.shape)[0]
                print("Number of samples: ", total_size)

                for epoch in range(epochs):
                        cost_ = 0
                        i = 0
			temp_sample = sample
			temp_label = label

                        #np.random.seed(epoch%1000)
                        #np.random.shuffle(temp_sample)

			#np.random.seed(epoch%1000)
                        #np.random.shuffle(temp_label)

			#print(temp_label)
                        #print("\nEPOCH: " + str(epoch))
                        #print("\n\n\nFC for samples 0 and 1")
                        #print(fc.eval({x: sample[0].reshape((1, 15000))}) - fc.eval({x: sample[1].reshape((1, 15000))}))
                        #print("\n\n\n")

                        #print("\n\n\nConv7: ")
                        #print(conv7.eval({x: sample[0].reshape((1, 15000))}) - conv7.eval({x: sample[1].reshape((1, 15000))}))
                        #print("\n\n\n")

                        #print("\n\n\nConv1: ")
                        #print(conv1.eval({x: sample[0].reshape((1, 15000))}) - conv1.eval({x: sample[1].reshape((1, 15000))}))
                        #print("\n\n\n")

                        #print("\n\n\nInput: ")
                        #print(inp.eval({x: sample[0].reshape((1, 15000))}) - inp.eval({x: sample[1].reshape((1, 15000))}))
			#print("\n\n\n")

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

		#print("\n\n\nFC for samples 0 and 1")
		#print(fc.eval({x: sample[0].reshape((1, 15000))}) - fc.eval({x: sample[1].reshape((1, 15000))}))
                #print("\n\n\n")

                #print("\n\n\nConv7: ")
 		#print(conv7.eval({x: sample[0].reshape((1, 15000))}) - conv7.eval({x: sample[1].reshape((1, 15000))}))
                #print(fc.eval({x: sample[1].reshape((1, 15000))}))
                #print("\n\n\n")

                print("Optimization complete...\n")
                print(prediction.eval({x: sample}))
                print(label)

                print("MSE for Q: ",  mse_q.eval({x: sample, y: label}))
                print("MSE for s1z: ", mse_s1z.eval({x: sample, y: label}))
                print("MSE for s2z: ",  mse_s2z.eval({x: sample, y: label}))

                correct = (tf.abs(tf.subtract(prediction, y)) < epsilon) #see if the difference is less than the threshold
                correct = tf.cast(correct, tf.float32)         #convert boolean tensor to float32
                accuracy = tf.reduce_mean(correct, axis=None)

                print('Training set accuracy: ', (accuracy.eval({x: sample, y: label})))
                print('Test set accuracy: ', (accuracy.eval({x: test_samples, y: test_labels})))


model(x)





