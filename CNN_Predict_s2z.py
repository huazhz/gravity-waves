import numpy as np
import random
import h5py as h5
import tensorflow as tf


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
        labels = np.transpose(labels)
        #print(labels[:,1])

        return dset, labels

# dset, labels = generate_data('train.h5')
# print(dset.shape)
# print(dset)
# print('\n')
# print(labels.shape)
# print(labels)

######## HELPER FUNCTIONS ########
def weight(name, shape):
        return tf.get_variable(name, shape=shape, initializer = tf.contrib.layers.xavier_initializer())

def bias(name, shape):
        return tf.Variable(tf.random_normal(shape), name=name)

def conv(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')

def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1, 5, 1, 1], strides=[1, 5, 1, 1], padding = 'SAME')


######## CNN MODEL ########
x = tf.placeholder(tf.float32, shape = [None, 15000])
y = tf.placeholder(tf.float32, shape = [None, 3])

def model(x):
        #Reshape input
        inp = tf.reshape(x, [-1, 15000, 1, 1])

        #Convolutional layer 
        w_conv1 = weight('w_conv1', [16, 1, 1, 64])
        b_conv1 = bias('b_conv1', [64])
        conv1 = tf.nn.relu(conv(inp, w_conv1) + b_conv1)

        #Max pool 1
        conv1 = maxPool(conv1)

        #Convolutional layer 2
        w_conv2 = weight('w_conv2', [16, 1, 64, 128])
        b_conv2 = bias('b_conv2', [128])
        conv2 = tf.nn.relu(conv(conv1, w_conv2) + b_conv2)

        #Max pool 2
        conv2 = maxPool(conv2)

        #Convolutional layer 3
        w_conv3 = weight('w_conv3', [16, 1, 128, 256])
        b_conv3 = bias('b_conv3', [256])
        conv3 = tf.nn.relu(conv(conv2, w_conv3) + b_conv3)

        #Max pool 3
        conv3 = maxPool(conv3)

        #Flatten
        flat = tf.reshape(conv3, [-1, 120 * 256])

        #Fully connected layer
        w_fc1 = weight('w_fc1', [120 * 256, 1024])
        b_fc1 = bias('b_fc1', [1024])
        fc = tf.nn.relu(tf.matmul(flat, w_fc1) + b_fc1)

        #Output
        w_fc2 = weight('w_f2', [1024, 3])
        b_fc2 = bias('b_fc2', [3])
        prediction = tf.matmul(fc, w_fc2) + b_fc2

        #return out


######## TRAIN MODEL ########
#def train(x):
        epochs = 150
        epsilon = .001
        #prediction = model(x)
        cost = (tf.losses.mean_squared_error(prediction, y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                sample, label = process_data('train50.h5')
                print(sample.shape, label.shape)
                test_samples, test_labels = process_data('test.h5')


                graph_cost = []
                graph_epoch = []

                for epoch in range(epochs):
                        _, c = sess.run([optimizer, cost], feed_dict = {x: sample, y: label})
                        if epoch % 10 == 0:
                                graph_epoch.append(epoch)
                                graph_cost.append(c)
                                print(str(epoch + 10) + " out of " + str(epochs) + " completed. Loss: " + str(c))


                print(w_fc2.eval())
                print("Optimization complete...\n")
                print(prediction.eval({x: sample}))
                print(label)

                correct = (tf.abs(tf.subtract(prediction, y)) < epsilon) #see if the difference is less than the threshold
                correct = tf.cast(correct, tf.float32)         #convert boolean tensor to float32
                accuracy = tf.reduce_mean(correct, axis=None)

                print('Training set accuracy: ', (accuracy.eval({x: sample, y: label})))
                print('Test set accuracy: ', (accuracy.eval({x: test_samples, y: test_labels})))


model(x)

