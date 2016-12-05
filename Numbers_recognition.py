import numpy as np
import tensorflow as tf
import cv2
import sklearn.cluster
import CNN_number_classifier
#from Sliding_window import sliding_window
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
    for start_x in range(0,img.shape[1],step):
        end_x=min(start_x+x_frame_size,img.shape[1])
        for start_y in range(0, img.shape[0], step):
            end_y = min(start_y + y_frame_size, img.shape[0])
            cut_img=img[start_y:end_y,start_x:end_x]
            img_to_classify=cv2.resize(cut_img,frame_size)
            prediction=predict(img_to_classify)
            #if prediction > thresh:
                #cv2.imwrite('1.jpg', img_to_classify * 255)
            res+=[(prediction,start_x,end_x,start_y,end_y)]
    return res
def create_predict_func(classifier):
    func=lambda X:classifier.predict(cv2.resize(X,(28,28)))#[0,number_to_rec]
    return func

def main():
    path='../../data2/image%d.gif.jpg'
    classifier=CNN_number_classifier.CNN_number_classifier()
    classifier.restore('MNIST_sess')

    size=(40,70)
    predict=create_predict_func(classifier)
    thresh=0.999
    number=6
    i=0
    img = cv2.imread(path % i)
    bgr=img.copy()
    m=img.shape[0]
    n=img.shape[1]
    cv2.namedWindow('bgr')
    #cv2.createTrackbar('s0','bgr',0,n-1,dummy)
    #cv2.createTrackbar('e0', 'bgr', 0, n - 1, dummy)


    while True:
        c=cv2.waitKey(30)
        if c==ord('q'):
            break
        if c == ord('n'):
            i+=1


            img=cv2.imread(path%i)
            img=255-img
        #s0=cv2.getTrackbarPos('s0','bgr')
        #e0 = cv2.getTrackbarPos('e0', 'bgr')
            gray=img[:,:,0]
        #x_positions=np.sum(gray,0)

        #cv2.kmeans(x_positions,6,)
            gray = cv2.dilate(gray,np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8))
        #res=sklearn.cluster.k_means(extractFeatures2(gray),7,max_iter=1000)
        #print res
        #bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            bgr = img.copy()
            res=sliding_window(predict,gray,3,size)
        #delta = time() - st
        #print delta, res
            for item in res:
                if item[0][0,number]<thresh:
                    continue
                cv2.rectangle(bgr,(item[1],item[3]),(item[2],item[4]),(255,0,255))
        cv2.imshow('bgr',bgr)

        #for k in range(res[0].shape[0]):
            #r = (int(res[0][k,0]), int(res[0][k, 1]))
            #cv2.circle(bgr, r, 2, (255, 0, 255))

        #cv2.line(bgr,(s0,0),(s0,m),(255,0,255))
        #cv2.line(bgr, (e0, 0), (e0, m), (255, 0, 255))


if __name__=='__main__':
    main()