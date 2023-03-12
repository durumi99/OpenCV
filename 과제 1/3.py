import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print('Camera open failed')
    exit()

print('Frame width : ', int(cap.get(cv.CAP_PROP_FRAME_WIDTH)))
print('Frame height : ', int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))
print('Frame count : ', int(cap.get(cv.CAP_PROP_FRAME_COUNT)))

w = round(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv.CAP_PROP_FPS)

fourcc = cv.VideoWriter_fourcc(*'DIVX')
delay = round(1000 / fps)

outputVideo = cv.VideoWriter('output.avi',fourcc,fps,(w,h),isColor=False)
if not outputVideo.isOpened():
    print('File open failed!')
    exit()

prev_time = 0
avg = -1
printInversed = False

while True :
    ret, frame = cap.read()
    if not ret:
        break

    if avg != -1 and abs(avg - np.mean(frame,dtype=np.int32)) > 30 :
        printInversed = True
        prev_time = cap.get(cv.CAP_PROP_POS_MSEC)

    if printInversed == True : 
        if cap.get(cv.CAP_PROP_POS_MSEC) - prev_time < 3000 :
            frame = ~frame
        else :
            printInversed = False
            
    avg = np.mean(frame,dtype=np.int32)
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    outputVideo.write(gray)
    cv.imshow('gray',gray)
    
    if cv.waitKey(delay) == 27:
        break

cv.destroyAllWindows()