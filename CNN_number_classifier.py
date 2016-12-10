import tensorflow as tf
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data
#mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

import cv2
class CNN_number_classifier:
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 60, 40])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
        x_image = tf.reshape(self.x, [-1, 60, 40, 1])

        W_conv1 = self.weight_variable([7, 7, 1, 32])
        b_conv1 = self.bias_variable([32])

        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        W_fc1 = self.weight_variable([15 * 10 * 64, 1024])
        b_fc1 = self.bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 15 * 10 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        W_fc2 = self.weight_variable([1024, 10])
        b_fc2 = self.bias_variable([10])
        self.y_conv = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        #cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_conv), reduction_indices=[1]))

        #self.y_conv = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.l2_loss=tf.reduce_mean((self.y_conv-self.y_)**2*((2*self.y_-1)**2))
        #self.mean_sq_err = tf.nn.l2_loss(self.y_conv - self.y_)
        #correct_prediction = tf.equal((tf.sign(self.y_conv - 0.5) + 1) / 2., self.y_)
        # tf.sign()
        #self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.train_step = tf.train.AdamOptimizer(1e-2).minimize(self.l2_loss)
        self.sess= tf.InteractiveSession()
        self.sess.run(tf.initialize_all_variables())
    def fit(self,X,Y,n=2000):
        for i in range(n):
            # batch = mnist.train.next_batch(50)
            batch = np.random.randint(X.shape[0], size=50)
            batch_data = X[batch]

            batch_labels = Y[batch]

            self.train_step.run(feed_dict={self.x: batch_data, self.y_: batch_labels, self.keep_prob: 0.5})
            if i%100==0:
                print i

            if i % 100 == 0:
                train_accuracy = self.l2_loss.eval(feed_dict={
                    self.x: batch_data, self.y_: batch_labels, self.keep_prob: 1.0})
                print train_accuracy



    def predict(self,X):
        return self.y_conv.eval(feed_dict={
            self.x: X.reshape(-1, 60, 40), self.keep_prob: 1.0})

    def restore(self,path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

def train():
    classifier=CNN_number_classifier()
    dataX=np.reshape(mnist.train.images,(-1,28,28))
    while True:
        c=cv2.waitKey(30)
        if c==ord('q'):
            break
        img=dataX[0]
        cv2.imshow('img',img)
    classifier.fit(dataX,mnist.train.labels,1000)
    classifier.save('MNIST_sess2')
if __name__=='__main__':
    train()