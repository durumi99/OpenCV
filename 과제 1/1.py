import cv2 as cv
import numpy as np

def saturated(value) :
    if value > 255:
        value = 255
    elif value < 0 :
        value = 0

    return value

def brightness():
    img = cv.imread('sample.jpg',cv.IMREAD_GRAYSCALE)

    if img is None :
        print('Image Load failed!')
        return
    avg = np.mean(img,dtype=np.int32)

    dst = np.empty(img.shape, dtype=img.dtype)
    
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y,x] < avg :
                dst[y,x] = 0
            else :
                dst[y,x] = img[y,x]

    # dst[img >= avg] = img[img >= avg]
    # dst[img < avg] = 0

    cv.imshow('image',img)
    cv.imshow('dst',dst) 
    cv.imwrite('output.jpg',dst)
    cv.waitKey()
    cv.destroyAllWindows()

brightness()