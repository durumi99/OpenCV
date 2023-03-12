import cv2 as cv

def hough_circles(src):
  blurred = cv.blur(src, (5, 5))
  circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, 1, 30, param1=250, param2=16, minRadius=10, maxRadius=16)
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
  src_bin = cv.adaptiveThreshold(src2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 5)
  src_bin = cv.morphologyEx(src_bin, cv.MORPH_OPEN, None)

  cnt, labels, stats, centroids = cv.connectedComponentsWithStats(src_bin)

  dst = cv.cvtColor(src2, cv.COLOR_GRAY2BGR)
  circle_cnt = 0

  for i in range(1, cnt):
    (x, y, w, h, area) = stats[i]
    
    if area < 20000 or area > 80000:
      continue

    pt1 = (x, y)
    pt2 = (x + w, y + h)

    cv.rectangle(dst, pt1, pt2, (0, 255, 255))
    
    # cv.imshow('src_bin',src_bin)
    # cv.imshow('dst', dst)

    # cv.waitKey()
    # cv.destroyAllWindows()

    tmp = src2[y:y+h, x:x+w]
    
    circle_cnt = hough_circles(tmp)
  
  return circle_cnt


for i in range(1,12):
  filename = 'img5_{}.png'.format(i)
  print(filename)
  src = cv.imread(filename, cv.IMREAD_COLOR)
  
  src2 = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
  res = labeling_stats(src2)
  print(res)
