import cv2 as cv
import numpy as np

def saturated(value) :
    if value > 255:
        value = 255
    elif value < 0 :
        value = 0

    return value

def contrast():
    img = cv.imread('sample.jpg',cv.IMREAD_GRAYSCALE)

    if img is None :
        print('Image Load failed!')
        return
    avg = np.mean(img,dtype=np.int32)
    alpha = 2.0
    dst = cv.convertScaleAbs(img,alpha= 1 + alpha, beta= - avg*alpha)
    
    dst2 = np.empty(img.shape, dtype=img.dtype)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            dst2[y,x] = saturated(img[y,x] + (img[y,x]- avg)*alpha)


    cv.imshow('image',img)
    cv.imshow('dst',dst)
    cv.imshow('dst2',dst2)
    cv.imwrite('contrast.jpg',dst2)
    cv.waitKey()
    cv.destroyAllWindows()

contrast()