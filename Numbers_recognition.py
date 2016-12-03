import numpy as np
import tensorflow as tf
import cv2
def dummy(x):
    pass
def main():
    path='../../data2/image%d.gif.jpg'

    i=0
    img = cv2.imread(path % i)
    m=img.shape[0]
    n=img.shape[1]
    cv2.namedWindow('bgr')
    cv2.createTrackbar('s0','bgr',0,n-1,dummy)
    cv2.createTrackbar('e0', 'bgr', 0, n - 1, dummy)
    while True:
        c=cv2.waitKey(30)
        if c==ord('q'):
            break
        if c == ord('n'):
            i+=1
        img=cv2.imread(path%i)
        img=255-img
        s0=cv2.getTrackbarPos('s0','bgr')
        e0 = cv2.getTrackbarPos('e0', 'bgr')
        bgr=img.copy()
        #bgr=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        cv2.line(bgr,(s0,0),(s0,m),(255,0,255))
        cv2.line(bgr, (e0, 0), (e0, m), (255, 0, 255))
        cv2.imshow('bgr',bgr)

if __name__=='__main__':
    main()