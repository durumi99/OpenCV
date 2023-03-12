import cv2 as cv
import numpy as np

def hough_circles(src):
  alpha = 0.216
  src = np.clip(src + (src - 128.)*alpha, 0, 255).astype(np.uint8)

  circles = cv.HoughCircles(src, cv.HOUGH_GRADIENT, 1, 20, param1=200, param2 = 15, minRadius=9, maxRadius=14)
  dst = cv.cvtColor(src, cv.COLOR_GRAY2BGR)

  if circles is not None :
    for i in range(circles.shape[1]):
      cx, cy, radius = circles[0][i]
      
      cv.circle(dst, (round(cx), round(cy)), round(radius), (0, 0, 255), 2, cv.LINE_AA)
    
    # cv.imshow('dst', dst)
    # cv.waitKey()
    # cv.destroyAllWindows()
    
    return circles.shape[1]

  return 0

def labeling_stats(src2):
  src_bin = cv.adaptiveThreshold(src2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 191, 5)
  src_bin = cv.morphologyEx(src_bin, cv.MORPH_OPEN, None, src_bin,(-1, -1), 2)
  
  cnt, labels, stats, centroids = cv.connectedComponentsWithStats(src_bin)

  dst = cv.cvtColor(src2, cv.COLOR_GRAY2BGR)
  arr = []

  for i in range(1, cnt):
    (x, y, w, h, area) = stats[i]

    if area < 5000 or area > 20000 :
      continue

    pt1 = (x, y)
    pt2 = (x + w, y + h)

    cv.rectangle(dst, pt1, pt2, (0, 255, 255))
  
    tmp = src2[y:y+h, x:x+w]
    circle_cnt = hough_circles(tmp)
    arr.append(circle_cnt)

  # cv.imshow('src_bin',src_bin)
  # cv.imshow('dst', dst)
  # cv.waitKey()
  # cv.destroyAllWindows()
  
  return arr

for i in range(1, 7):
  filename = 'img4_{}.png'.format(i)
  print(filename)
  src = cv.imread(filename, cv.IMREAD_COLOR)                                                                   
  src2 = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
  arr = labeling_stats(src2)
  arr.sort()
  print(arr)