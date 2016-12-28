import cv2
import numpy as np
import random

def get_matching_img(cl_result,img_size,window_size):
    matching_img=np.zeros((img_size[1]-window_size[1],img_size[0]-window_size[0],10))
    for item in cl_result:
        matching_img[item[3],item[1]]=item[0][0]
    return matching_img

def create_matching_images(train_s_imgs,train_s_labels,img_size,window_size,predict,sliding_window,step=4):
    matching_imgs = np.zeros((len(train_s_imgs),img_size[1]-window_size[1],img_size[0]-window_size[0],10),np.float32)
    matching_labels = np.zeros((len(train_s_imgs),10,6),np.float32)
    p_imgs=np.load('matching_imgs5_575.npy')
    p_labels=np.load('matching_labels5_575.npy')
    matching_imgs[:575]=p_imgs[:575]
    matching_labels[:575]=p_labels[:575]

    for i in range(575,len(train_s_imgs)):
        img = train_s_imgs[i]
        gray = img
        # elem=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        # gray=cv2.dilate(img,elem)
        res = sliding_window(predict, gray.astype(np.float32) / 255, step, window_size)
        matching_img = get_matching_img(res, img_size, window_size)
        matching_imgs[i] = matching_img
        label = np.zeros((10, 6))
        for j in range(6):
            label[train_s_labels[i][j]][j] = 1
        matching_labels[i] = label
        print i

    #matching_imgs = np.array(matching_imgs)
    #matching_labels = np.array(matching_labels)
    return matching_imgs,matching_labels

def split(s, delims):
    def splitLst(slst, delims):
        if len(delims) == 0:
            return slst
        r = []
        for s in slst:
            r += s.split(delims[0])
        return splitLst(r, delims[1:])
    return filter(lambda x : x != '', splitLst([s], delims))

def read_data(path,img_path):
    f = open(path)
    lines = f.readlines()
    source_imgs = []
    source_labels = []
    source_positions=[]
    for line in lines:
        positions = []
        nums = []
        line_data = split(line, '() ,\n')
        i = int(line_data[0])
        positions += [(int(line_data[1]), int(line_data[2]))]
        positions += [(int(line_data[3]), int(line_data[4]))]
        positions += [(int(line_data[5]), int(line_data[6]))]
        positions += [(int(line_data[7]), int(line_data[8]))]
        positions += [(int(line_data[9]), int(line_data[10]))]
        positions += [(int(line_data[11]), int(line_data[12]))]
        nums += [int(line_data[13])]
        nums += [int(line_data[14])]
        nums += [int(line_data[15])]
        nums += [int(line_data[16])]
        nums += [int(line_data[17])]
        nums += [int(line_data[18])]
        img = 255 - cv2.imread(img_path % i)[:, :, 0]
        source_positions+=[positions]
        source_labels+=[nums]
        source_imgs+=[img]
    return np.array(source_imgs,np.float32)/255,np.array(source_positions),np.array(source_labels)

def prepare_imgs(imgs,prepare):
    prepared_imgs=prepare(imgs)
    return prepared_imgs

def divide_source_data(source_imgs,source_positions,source_labels,ratio,mode='w'):
    n=len(source_imgs)
    if mode=='w':

        train_ids=random.sample(range(n),int(ratio*n))
        f=open('partition2','w')
        f.writelines([str(train_ids)])
    else:
        f=open('partition2','r')
        line=f.readline()
        train_ids=[int(s) for s in line[1:-1].split(', ')]
    test_ids=list(set(range(n))-set(train_ids))
    train_imgs=source_imgs[train_ids]
    train_labels=source_labels[train_ids]
    train_positions = source_positions[train_ids]
    test_imgs=source_imgs[test_ids]
    test_labels=source_labels[test_ids]
    test_positions = source_positions[test_ids]
    return train_imgs,train_positions,train_labels,test_imgs,test_positions,test_labels

def create_data(source_imgs,source_positions,source_labels,size=(40,60),neg_num=3):

    imgs=[]
    labels=[]
    for i in range(len(source_imgs)):
        positions = source_positions[i]
        img=source_imgs[i]
        nums=source_labels[i]

        #gray = cv2.GaussianBlur(img, (3, 3), -1)
        #elem=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        gray=img
        #gray=cv2.dilate(img,elem)

        loc_imgs,loc_labels=create_imgs_and_labels(gray,positions,nums,size,neg_num)
        imgs+=loc_imgs
        labels+=loc_labels
        pass
    imgs=np.array(imgs,np.float32)#/255
    labels=np.array(labels)
    return imgs,labels

def create_imgs_and_labels(img,positions,nums,size,neg_num,offset=3):
    imgs=[]

    labels=[]
    for i in range(len(positions)):
        pos=positions[i]
        num=nums[i]
        x=pos[0]-size[0]/2
        y=pos[1]-size[1]/2
        #gray=cv2.dilate(img[y:y+size[1],x:x+size[0]],np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8))
        #gray = cv2.GaussianBlur(img[y:y + size[1], x:x + size[0]], (3, 3), -1)
        imgs+=[img[y:y + size[1], x:x + size[0]]]
        labels+=[[0 for i in range(10)]]
        labels[-1][num]=1
    for i in range(neg_num):
        x=random.randint(0,img.shape[1]-size[0])
        y=random.randint(0,img.shape[0]-size[1])
        #gray=cv2.dilate(img[y:y+size[1],x:x+size[0]],np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8))
        #gray = cv2.GaussianBlur(img[y:y + size[1], x:x + size[0]],(3,3),-1)

        imgs+=[img[y:y + size[1], x:x + size[0]]]
        labels+=[[0 for i in range(10)]]
        for i in range(len(positions)):

            if abs(x+size[0]/2-positions[i][0])<size[0]/2 and abs(y+size[1]/2-positions[i][1])<size[1]/2:
                if abs(x+size[0]/2-positions[i][0])<offset and abs(y+size[1]/2-positions[i][1])<offset:
                    labels[-1][nums[i]]=1
                else:
                    labels[-1][nums[i]] = 0#.5
            else:
                labels[-1][nums[i]]=0
    return imgs,labels

def get_skeleton_image(source_img):
    img=source_img.copy()
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)

    #ret,img = cv2.threshold(img,127,255,0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    img=cv2.dilate(img,element)
    done = False

    while not done:
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
    return skel

def main():
    imgs,labels=create_data('pos_and_lab.txt','../../data2/image%d.gif.jpg')
    num=6
    for i in range(labels.shape[0]):
        if labels[i][num]==1:
            img=imgs[i]
            break

    while True:
        c=cv2.waitKey(30)
        if c==ord('q'):
            break
        cv2.imshow('img',img)
def save_data():
    imgs,labels=create_data('pos_and_lab.txt','../../data2/image%d.gif.jpg',(40,60),0)
    np.savez('imgs.npz',imgs)
    np.savez('labels.npz',labels)

if __name__=='__main__':
    save_data()
    #main()