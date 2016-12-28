import tensorflow as tf
import numpy as np
class Digit_localizer:
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    def __init__(self,source_size):
        self.source_size=source_size

        self.x=tf.placeholder(tf.float32, shape=[None,source_size[1], source_size[0],10])
        self.y_=tf.placeholder(tf.float32,shape=[None,10,6])
        #reduced4x=tf.nn.max_pool(self.x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

        reduced8x=tf.nn.avg_pool(self.x, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')


        digit_maps_list=[]
        other_maps_list=[]
        for i in range(10):
            digit_map=reduced8x[:,:,:,i:i+1]
            other_map=tf.reduce_max(tf.concat(3,[reduced8x[:,:,:,:i],reduced8x[:,:,:,i+1:]]),3)
            other_map_s=other_map.get_shape()
            other_map=tf.reshape(other_map,(-1,int(other_map_s[1]),int(other_map_s[2]),1))
            digit_maps_list+=[digit_map]
            other_maps_list+=[other_map]
            pass
        self.digit_maps=tf.concat(3,digit_maps_list)
        self.other_maps=tf.concat(3,other_maps_list)
        self.map_shape=self.digit_maps.get_shape()[1:3]
        self.digit_maps_flat=tf.reshape(self.digit_maps,(-1,int(self.map_shape[0]*self.map_shape[1]),10))

        self.other_maps_flat=tf.reshape(self.digit_maps,(-1,int(self.map_shape[0]*self.map_shape[1]),10))


        self.w=self.weight_variable((2*int(self.map_shape[0]*self.map_shape[1]),6))
        self.b=self.bias_variable((6,))




        digit_values=[]
        for i in range(10):
            digit_map_flat=tf.concat(1,[self.digit_maps_flat[:,:,i],self.other_maps_flat[:,:,i]])
            digit_value=tf.matmul(digit_map_flat,self.w)+self.b
            digit_values+=[tf.reshape(digit_value,(-1,6,1))]
        self.digit_values=tf.concat(2,digit_values)
        digit_prob_list=[]
        for i in range(6):
            digit_probs=tf.nn.softmax(self.digit_values[:,i,:])
            digit_prob_list+=[tf.reshape(digit_probs,(-1,10,1))]
        self.digit_probs=tf.concat(2,digit_prob_list)
        self.mse = tf.reduce_mean(tf.pow(self.digit_probs - self.y_, 2))
        '''

        h_size=10
        self.w=self.weight_variable((2*int(self.map_shape[0]*self.map_shape[1]),h_size))
        self.b=self.bias_variable((h_size,))

        self.w2=self.weight_variable((h_size,6))
        self.b2=self.bias_variable((6,))



        digit_values=[]
        for i in range(10):
            digit_map_flat=tf.concat(1,[self.digit_maps_flat[:,:,i],self.other_maps_flat[:,:,i]])
            digit_value=tf.nn.relu(tf.matmul(digit_map_flat,self.w)+self.b)
            digit_value2=tf.matmul(digit_value,self.w2)+self.b2

            digit_values+=[tf.reshape(digit_value2,(-1,6,1))]
        self.digit_values=tf.concat(2,digit_values)
        digit_prob_list=[]
        for i in range(6):
            digit_probs=tf.nn.softmax(self.digit_values[:,i,:])
            digit_prob_list+=[tf.reshape(digit_probs,(-1,10,1))]
        self.digit_probs=tf.concat(2,digit_prob_list)
        self.mse = tf.reduce_mean(tf.pow(self.digit_probs - self.y_, 2))
        '''


        self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.digit_probs), reduction_indices=[1]))
        self.correct_prediction = tf.equal(tf.argmax(self.digit_probs, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.train_step = tf.train.AdamOptimizer(0.001).minimize(self.mse)


        self.sess=tf.Session()
        self.sess.run(tf.initialize_all_variables())

    def fit(self,X,Y,test_X,test_Y,n=2000):
        for i in range(n):
            # batch = mnist.train.next_batch(50)
            batch = np.random.randint(X.shape[0], size=50)
            batch_data = X[batch]

            batch_labels = Y[batch]

            if i%100==1:
                print i

            if i % 100 == 1:
                #s = self.y_conv.eval(feed_dict={self.x : batch_data, self.keep_prob: 1.0})
                train_loss = self.accuracy.eval(session=self.sess,feed_dict={
                    self.x: batch_data, self.y_: batch_labels})
                test_loss=''
                if test_X!=None and test_Y!=None:

                    test_loss = self.accuracy.eval(session=self.sess,feed_dict={
                        self.x: test_X, self.y_: test_Y})

                print train_loss,' ',test_loss

            self.train_step.run(session=self.sess,feed_dict={self.x: batch_data, self.y_: batch_labels})
        pass

    def predict(self,X):
        predictions=self.digit_probs.eval(session=self.sess,feed_dict={
            self.x: X.reshape(-1, self.source_size[1], self.source_size[0],10)})
        res=np.zeros((len(predictions),6))-1
        for i in range(len(predictions)):
            res[i]=np.argmax(predictions[i],0)
        return res


    def restore(self,path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)

    def save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)
