import cv2
import numpy as np
import random

def split(s, delims):
    def splitLst(slst, delims):
        if len(delims) == 0:
            return slst
        r = []
        for s in slst:
            r += s.split(delims[0])
        return splitLst(r, delims[1:])
    return filter(lambda x : x != '', splitLst([s], delims))

def create_data(path,img_path,size=(40,60),neg_num=3):
    f=open(path)
    lines=f.readlines()
    imgs=[]
    labels=[]
    for line in lines:
        positions = []
        nums = []
        line_data=split(line, '() ,\n')
        i=int(line_data[0])
        positions+=[(int(line_data[1]),int(line_data[2]))]
        positions += [(int(line_data[3]), int(line_data[4]))]
        positions += [(int(line_data[5]), int(line_data[6]))]
        positions += [(int(line_data[7]), int(line_data[8]))]
        positions += [(int(line_data[9]), int(line_data[10]))]
        positions += [(int(line_data[11]), int(line_data[12]))]
        nums+=[int(line_data[13])]
        nums += [int(line_data[14])]
        nums += [int(line_data[15])]
        nums += [int(line_data[16])]
        nums += [int(line_data[17])]
        nums += [int(line_data[18])]
        img=255-cv2.imread(img_path%i)[:,:,0]
        loc_imgs,loc_labels=create_imgs_and_labels(img,positions,nums,size,neg_num)
        imgs+=loc_imgs
        labels+=loc_labels
        pass
    imgs=np.array(imgs,np.float32)/255
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
        gray = cv2.GaussianBlur(img[y:y + size[1], x:x + size[0]], (3, 3), -1)
        imgs+=[gray]
        labels+=[[0 for i in range(10)]]
        labels[-1][num]=1
    for i in range(neg_num):
        x=random.randint(0,img.shape[1]-size[0])
        y=random.randint(0,img.shape[0]-size[1])
        #gray=cv2.dilate(img[y:y+size[1],x:x+size[0]],np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8))
        gray = cv2.GaussianBlur(img[y:y + size[1], x:x + size[0]],(3,3),-1)
        imgs+=[gray]
        labels+=[[0 for i in range(10)]]
        for i in range(len(positions)):

            if abs(x+size[0]/2-positions[i][0])<size[0]/2 and abs(y+size[1]/2-positions[i][1])<size[1]/2:
                if abs(x+size[0]/2-positions[i][0])<offset and abs(y+size[1]/2-positions[i][1])<offset:
                    labels[-1][nums[i]]=1
                else:
                    labels[-1][nums[i]] = 0.5
            else:
                labels[-1][nums[i]]=0
    return imgs,labels


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