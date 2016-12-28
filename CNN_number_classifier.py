import tensorflow as tf
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data
#mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

import cv2
class CNN_number_classifier:
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        var=tf.Variable(initial)
        self.var_list+=[var]
        return var

    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        var=tf.Variable(initial)
        self.var_list+=[var]
        return var

    def conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self,x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def __init__(self,size=(40,60)):
        self.var_list=[]
        self.size = size
        self.x = tf.placeholder(tf.float32, shape=[None, size[1], size[0]])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])
        x_image = tf.reshape(self.x, [-1, size[1], size[0], 1])


        '''
        W_conv0 = self.weight_variable([3, 3, 1, 16])
        b_conv0 = self.bias_variable([16])

        h_conv0 = tf.nn.relu(self.conv2d(x_image, W_conv0) + b_conv0)

        W_conv1 = self.weight_variable([5, 5, 16, 32])
        b_conv1 = self.bias_variable([32])

        h_conv1 = tf.nn.relu(self.conv2d(h_conv0, W_conv1) + b_conv1)

        '''

        W_conv1 = self.weight_variable([5, 5, 1, 32])

        b_conv1 = self.bias_variable([32])

        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        self.W_fc1 = self.weight_variable([size[0]*size[1] * 64/16, 1024])
        self.b_fc1 = self.bias_variable([1024])

        self.h_pool2_flat = tf.reshape(h_pool2, [-1, size[0]*size[1] * 64/16])
        h_fc1 = tf.nn.relu(tf.matmul(self.h_pool2_flat, self.W_fc1) + self.b_fc1)

        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        self.W_fc2 = self.weight_variable([1024, 10])
        self.b_fc2 = self.bias_variable([10])
        self.y_conv = tf.nn.sigmoid(tf.matmul(h_fc1_drop, self.W_fc2) + self.b_fc2)

        #cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y_conv), reduction_indices=[1]))

        #self.y_conv = tf.nn.sigmoid(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        #correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        #self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.l2_loss=tf.reduce_mean(tf.reduce_sum(tf.pow((self.y_conv-self.y_)*(tf.pow((2*self.y_-1),2)),2),reduction_indices=[1]))
        #self.l2_loss = tf.reduce_mean(tf.reduce_sum(tf.pow(self.y_conv - self.y_, 2), reduction_indices=[1]))

        #self.l2_loss = tf.reduce_mean((2*(self.y_conv - self.y_))**(2))
        #correct_prediction = tf.equal((tf.sign(self.y_conv - 0.5) + 1) / 2., self.y_)
        # tf.sign()
        #self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.train_step = tf.train.AdamOptimizer(0.0001).minimize(self.l2_loss)
        self.sess= tf.Session()
        self.sess.run(tf.initialize_all_variables())
    def fit(self,X,Y,test_X,test_Y,n=2000,prob_space=None):
        if prob_space==None:
            prob_space = np.linspace(0.5, 0.5, n)
        for i in range(n):
            # batch = mnist.train.next_batch(50)
            batch = np.random.randint(X.shape[0], size=50)
            batch_data = X[batch]

            batch_labels = Y[batch]

            if i%100==1:
                print i

            if i % 100 == 1:
                #s = self.y_conv.eval(feed_dict={self.x : batch_data, self.keep_prob: 1.0})
                train_loss = self.l2_loss.eval(session=self.sess,feed_dict={
                    self.x: batch_data, self.y_: batch_labels, self.keep_prob: 1.0})
                test_loss=''
                if test_X!=None and test_Y!=None:
                    batch = np.random.randint(test_X.shape[0], size=100)
                    test_batch_data = test_X[batch]

                    test_batch_labels = test_Y[batch]
                    test_loss = self.l2_loss.eval(session=self.sess,feed_dict={
                        self.x: test_batch_data, self.y_: test_batch_labels, self.keep_prob: 1.0})

                print train_loss,' ',test_loss

            self.train_step.run(session=self.sess,feed_dict={self.x: batch_data, self.y_: batch_labels, self.keep_prob: prob_space[i]})


    def predict(self,X):
        return self.y_conv.eval(session=self.sess,feed_dict={
            self.x: X.reshape(-1, self.size[1], self.size[0]), self.keep_prob: 1.0})

    def restore(self,path):
        saver = tf.train.Saver(self.var_list)
        saver.restore(self.sess, path)

    def save(self, path):
        saver = tf.train.Saver(self.var_list)
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