import numpy as np
import tensorflow as tf
import cv2
import sklearn.cluster
import CNN_number_classifier
import random
#from Sliding_window import sliding_window
from tensorflow.examples.tutorials.mnist import input_data
from Digit_localizer import Digit_localizer
from Create_data import create_data,divide_source_data,read_data,create_matching_images,get_matching_img,prepare_imgs
from Restore_img import Restorer


def get_mnist_dataset(size=(40,60)):
    mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
    imgs=[]
    for img in mnist.train.images:
        gray=np.reshape(img,(28,28))
        gray=cv2.resize(gray,size)
        imgs+=[gray]
    imgs=np.array(imgs)
    labels=mnist.train.labels
    return imgs,labels
def dummy(x):
    pass

def extractFeatures(img):
    samples = np.sum([img != 0])
    features = np.zeros((samples, 2))
    k = 0
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            if img.item((y, x)) != 0:
                features[k, 0] = x
                features[k, 1] = y#img.shape[0] / 2
                k += 1
    return features

def extractFeatures2(img):
    samples = np.sum([img != 0])
    features = np.zeros((samples, 2))
    k = 0
    for x in range(img.shape[1]):
        for y in range(np.sum(img[:,x]/255)):
                features[k, 0] = x
                features[k, 1] = y#img.shape[0] / 2
                k += 1
    return features

def sliding_window(predict,img,step,frame_size):
    res=[]
    x_frame_size,y_frame_size=frame_size
    for start_x in range(0,img.shape[1]-x_frame_size,step):
        end_x=start_x+x_frame_size
        for start_y in range(0, img.shape[0]-y_frame_size, step):
            end_y = start_y + y_frame_size
            cut_img=img[start_y:end_y,start_x:end_x]
            img_to_classify=cv2.resize(cut_img,frame_size)
            prediction=predict(img_to_classify)
            #if prediction > thresh:
                #cv2.imwrite('1.jpg', img_to_classify * 255)
            res+=[(prediction,start_x,end_x,start_y,end_y)]
    return res

def divide_data(imgs,labels,ratio):
    n=len(imgs)
    train_ids=random.sample(range(n),int(ratio*n))
    test_ids=list(set(range(n))-set(train_ids))
    train_imgs=imgs[train_ids]
    train_labels=labels[train_ids]
    test_imgs=imgs[test_ids]
    test_labels=labels[test_ids]
    return train_imgs,train_labels,test_imgs,test_labels

def create_predict_func(classifier,size):
    func=lambda X:classifier.predict(cv2.resize(X,size))#[0,number_to_rec]
    return func

def create_prepare_func(restorer):
    func=lambda X:restorer.predict(X.reshape(X.shape+(1,)))[:,:,:,0]
    return func

def reshape_images(imgs,size):
    res_imgs=[]
    for img in imgs:
        res_imgs+=[cv2.resize(img,size)]
    return np.array(res_imgs)




def main():
    #size = (28, 28)
    size = (24,36)
    window_size=(40,60)
    img_size=(200,100)
    #imgs,labels=get_mnist_dataset(size)
    path='../../data2/image%d.gif.jpg'

    classifier=CNN_number_classifier.CNN_number_classifier(size)
    source_imgs,source_positions,source_labels=read_data('pos_and_lab.txt','../../data2/image%d.gif.jpg')
    train_s_imgs,train_s_positions,train_s_labels,test_s_imgs,test_s_positions,test_s_labels=divide_source_data(source_imgs,source_positions,source_labels,0.9,mode='r')
    train_imgs,train_labels=create_data(train_s_imgs,train_s_positions,train_s_labels,neg_num=1)
    train_imgs=reshape_images(train_imgs,size)
    test_imgs,test_labels=create_data(test_s_imgs,test_s_positions,test_s_labels,neg_num=1)
    test_imgs=reshape_images(test_imgs,size)
    #train_imgs,train_labels,test_imgs,test_labels=divide_data(imgs,labels,0.9)
    '''
    while True:
        c=cv2.waitKey(30)
        if c==ord('q'):
            break
        cv2.imshow('img',imgs[1])
    '''
    #res0=classifier.predict(train_imgs[1])
    classifier.restore('sess11')
    iters=7102
    prob_space=np.linspace(0.95,0.5,iters)
    #classifier.fit(train_imgs,train_labels,None,None,iters,prob_space)
    #classifier.save('sess11')
    #res1=classifier.predict(imgs[1])

    predict=create_predict_func(classifier,size)

    matching_imgs,matching_labels=create_matching_images(train_s_imgs,train_s_labels,img_size,window_size,predict,sliding_window)
    np.save('matching_imgs', matching_imgs)
    np.save('matching_labels', matching_labels)

    #matching_imgs=np.load('matching_imgs.npy')
    #matching_labels=np.load('matching_labels.npy')


    digit_localizer=Digit_localizer((img_size[0]-window_size[0],img_size[1]-window_size[1]))
    digit_localizer.fit(matching_imgs,matching_labels,None,None,5000)




    thresh=0.9
    number=4
    i=0
    img = cv2.imread(path % i)
    bgr=img.copy()
    m=img.shape[0]
    n=img.shape[1]
    cv2.namedWindow('bgr')
    #cv2.createTrackbar('s0','bgr',0,n-1,dummy)
    #cv2.createTrackbar('e0', 'bgr', 0, n - 1, dummy)

    c=ord('n')
    while True:

        if c==ord('q'):
            break
        if c == ord('n'):
            i+=1


            img=test_s_imgs[i]
            gray=img
            elem=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
            gray=cv2.dilate(img,elem)
            #gray=cv2.erode(gray,elem)
            #gray = cv2.GaussianBlur(gray,(3,3),-1)
            bgr = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
            res=sliding_window(predict,gray.astype(np.float32)/255,4,window_size)
            for item in res:
                for num in range(10):
                    if item[0][0,num]>=thresh:
                        cv2.rectangle(bgr,(item[1],item[3]),(item[2],item[4]),(26*num,0,255-26*num))

                    continue
            #matching_img=get_matching_img(res,(img.shape[1],img.shape[0]),window_size)

            #digit_maps=digit_localizer.digit_maps.eval(feed_dict={digit_localizer.x:matching_img.reshape(-1,40,160,10)})
        cv2.imshow('bgr',bgr)
        c = cv2.waitKey(30)
def prepare_matching_imgs():
    #size = (28, 28)
    size = (32,44)
    window_size=(40,60)
    img_size=(200,100)
    step=2

    restorer=Restorer()
    restorer.restore('r_sess2')
    prepare_func=create_prepare_func(restorer)


    classifier=CNN_number_classifier.CNN_number_classifier(size)
    source_imgs,source_positions,source_labels=read_data('pos_and_lab.txt','../../data2/image%d.gif.jpg')

    #prepared_imgs=prepare_imgs(source_imgs,prepare_func)
    prepared_imgs=np.load('restored_imgs.npy')

    train_s_imgs,train_s_positions,train_s_labels,test_s_imgs,test_s_positions,test_s_labels=divide_source_data(prepared_imgs,source_positions,source_labels,0.9,mode='r')

    classifier.restore('sess18_32x44_s')

    predict=create_predict_func(classifier,size)

    matching_imgs,matching_labels=create_matching_images(train_s_imgs,train_s_labels,img_size,window_size,predict,sliding_window,step)
    np.save('matching_imgs5', matching_imgs)
    np.save('matching_labels5', matching_labels)

    matching_imgs,matching_labels=create_matching_images(test_s_imgs,test_s_labels,img_size,window_size,predict,sliding_window,step)
    np.save('test_matching_imgs5', matching_imgs)
    np.save('test_matching_labels5', matching_labels)
def train_restorer():
    n=10
    epochs=1000
    line_num=5
    thickness=2
    size=(100,100)
    keep_prob=0.6
    imgs=np.zeros((n,size[1],size[0]))
    source_imgs=np.zeros((n,size[1],size[0]))
    for i in range(n):
        img=imgs[i]
        for j in range(line_num):
            pt1=(random.randint(0,size[0]),random.randint(0,size[1]))
            pt2=(random.randint(0,size[0]),random.randint(0,size[1]))
            cv2.line(img,pt1,pt2,1.0,thickness)
        source_imgs[i]=img.copy()
        for x in range(size[0]):
            for y in range(size[1]):
                if img[y,x]==0:
                    continue
                r=random.random()
                img[y,x]=float(r<=keep_prob)
    imgs=imgs.reshape((n,size[1],size[0],1))
    source_imgs=source_imgs.reshape((n,size[1],size[0],1))
    restorer=Restorer()
    #restorer.restore('r_sess2')
    restorer.fit(imgs,source_imgs,None,None,epochs)
    restorer.save('r_sess2')

def prepare_restored_imgs():
    source_imgs, source_positions, source_labels = read_data('pos_and_lab.txt', '../../data2/image%d.gif.jpg')
    restorer=Restorer()
    restorer.restore('r_sess2')
    prepare_func = create_prepare_func(restorer)
    prepared_imgs=[]
    for i in range(len(source_imgs)):

        prepared_imgs += [prepare_imgs(np.array([source_imgs[i]]), prepare_func)[0]]

    restored_imgs=np.array(prepared_imgs)
    np.save('restored_imgs',restored_imgs)

def train_classifier():

    restorer=Restorer()
    restorer.restore('r_sess2')
    #size = (28, 28)
    size = (32,44)
    window_size=(40,60)
    img_size=(200,100)
    #imgs,labels=get_mnist_dataset(size)
    path='../../data2/image%d.gif.jpg'

    classifier=CNN_number_classifier.CNN_number_classifier(size)
    source_imgs,source_positions,source_labels=read_data('pos_and_lab.txt','../../data2/image%d.gif.jpg')
    #prepare_func=create_prepare_func(restorer)
    #prepared_imgs=prepare_imgs(source_imgs,prepare_func)
    prepared_imgs=np.load('restored_imgs.npy')
    train_s_imgs,train_s_positions,train_s_labels,test_s_imgs,test_s_positions,test_s_labels=divide_source_data(prepared_imgs,source_positions,source_labels,0.9,mode='r')
    train_imgs,train_labels=create_data(train_s_imgs,train_s_positions,train_s_labels,neg_num=1)
    train_imgs=reshape_images(train_imgs,size)
    test_imgs,test_labels=create_data(test_s_imgs,test_s_positions,test_s_labels,neg_num=1)
    test_imgs=reshape_images(test_imgs,size)


    classifier.restore('sess18_32x44_s')
    iters=7102
    prob_space=np.linspace(0.6,0.6,iters)
    classifier.fit(train_imgs,train_labels,test_imgs,test_labels,iters,prob_space)
    classifier.save('sess18_32x44_s')




def train_localizer():
    size = (32,44)
    img_size=(200,100)
    window_size=(40,60)
    img_size=(200,100)

    matching_imgs=np.load('matching_imgs5.npy')
    matching_labels=np.load('matching_labels5.npy')

    test_matching_imgs=np.load('test_matching_imgs5.npy')
    test_matching_labels=np.load('test_matching_labels5.npy')
    restorer=Restorer()
    classifier=CNN_number_classifier.CNN_number_classifier(size)
    digit_localizer=Digit_localizer((img_size[0]-window_size[0],img_size[1]-window_size[1]))

    digit_localizer.fit(matching_imgs,matching_labels,test_matching_imgs,test_matching_labels,50000)
    digit_localizer.save('loc_sess30_size32x44_as8')
    return


def test():
    #size = (28, 28)
    size = (32,44)
    window_size=(40,60)
    img_size=(200,100)
    #imgs,labels=get_mnist_dataset(size)
    #path='../../data2/image%d.gif.jpg'

    source_imgs,source_positions,source_labels=read_data('pos_and_lab.txt','../../data2/image%d.gif.jpg')
    train_s_imgs,train_s_positions,train_s_labels,test_s_imgs,test_s_positions,test_s_labels=divide_source_data(source_imgs,source_positions,source_labels,0.9,mode='r')
    #train_imgs,train_labels=create_data(train_s_imgs,train_s_positions,train_s_labels,neg_num=1)
    #train_imgs=reshape_images(train_imgs,size)
    #test_imgs,test_labels=create_data(test_s_imgs,test_s_positions,test_s_labels,neg_num=1)
    #test_imgs=reshape_images(test_imgs,size)
    #saver=tf.train.Saver()
    classifier=CNN_number_classifier.CNN_number_classifier(size)
    classifier.restore('sess13')
    print np.sum(classifier.W_fc1.eval(classifier.sess))
    digit_localizer=Digit_localizer((img_size[0]-window_size[0],img_size[1]-window_size[1]))
    digit_localizer.restore('loc_sess15_size32x44')
    print np.sum(classifier.W_fc1.eval(classifier.sess))


    predict=create_predict_func(classifier,size)






    for i in range(len(test_s_imgs)):
        img=test_s_imgs[i]
        label=test_s_labels[i]
        elem=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        gray=cv2.dilate(img,elem)
        res = sliding_window(predict, gray.astype(np.float32) / 255, 4, window_size)
        matching_img = get_matching_img(res, img_size, window_size)
        p_label=digit_localizer.predict(matching_img)
        print 'l = ',label,' p = ',p_label
        while True:
            c=cv2.waitKey(30)
            if c==ord('n'):
                break
            if c==ord('q'):
                break
            cv2.imshow('img',img)
        if c==ord('q'):
            break

def automatic_test():
    size = (32, 44)
    #size = (24,36)
    window_size=(40,60)
    img_size=(200,100)
    #imgs,labels=get_mnist_dataset(size)
    #path='../../data2/image%d.gif.jpg'

    source_imgs,source_positions,source_labels=read_data('pos_and_lab.txt','../../data2/image%d.gif.jpg')
    train_s_imgs,train_s_positions,train_s_labels,test_s_imgs,test_s_positions,test_s_labels=divide_source_data(source_imgs,source_positions,source_labels,0.9,mode='r')
    #train_imgs,train_labels=create_data(train_s_imgs,train_s_positions,train_s_labels,neg_num=1)
    #train_imgs=reshape_images(train_imgs,size)
    #test_imgs,test_labels=create_data(test_s_imgs,test_s_positions,test_s_labels,neg_num=1)
    #test_imgs=reshape_images(test_imgs,size)
    #saver=tf.train.Saver()
    classifier=CNN_number_classifier.CNN_number_classifier(size)
    classifier.restore('sess18_32x44_s')
    print np.sum(classifier.W_fc1.eval(classifier.sess))
    digit_localizer=Digit_localizer((img_size[0]-window_size[0],img_size[1]-window_size[1]))
    digit_localizer.restore('loc_sess27_size32x44_as8')
    print np.sum(classifier.W_fc1.eval(classifier.sess))


    predict=create_predict_func(classifier,size)





    correct_num=0
    for i in range(len(test_s_imgs)):
        img=test_s_imgs[i]
        label=test_s_labels[i]
        elem=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        gray=cv2.dilate(img,elem)
        res = sliding_window(predict, gray.astype(np.float32) / 255, 2, window_size)
        matching_img = get_matching_img(res, img_size, window_size)
        p_label=digit_localizer.predict(matching_img)
        err=False
        print i
        for j in range(6):
            if label[j]!=p_label[0][j]:
                err=True
                print 'l = ',label,' p = ',p_label
                break

        if not err:
            correct_num+=1

    print float(correct_num)/len(test_s_labels)




if __name__=='__main__':
    #test()
    #automatic_test()
    prepare_matching_imgs()
    #prepare_restored_imgs()
    #train_classifier()
    #train_localizer()
    #main()