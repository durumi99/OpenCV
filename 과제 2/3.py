import cv2 as cv

def color_split(src) :
  src_ycrcb = cv.cvtColor(src, cv.COLOR_BGR2YCrCb)
  ycrcb_planes = cv.split(src_ycrcb)
  tmp = cv.equalizeHist(ycrcb_planes[0])
  dst_ycrcb = cv.merge((tmp, ycrcb_planes[1], ycrcb_planes[2]))
  dst = cv.cvtColor(dst_ycrcb, cv.COLOR_YCrCb2BGR)
  src2 = cv.cvtColor(dst,cv.COLOR_BGR2GRAY)
  
  return src2


def hough_circles(src):
  blurred = cv.blur(src, (5, 5))
  circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, 1, 20, param1=150, param2=14, minRadius=7, maxRadius=15)
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
  src_bin = cv.adaptiveThreshold(src2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 5)
  src_bin = cv.morphologyEx(src_bin, cv.MORPH_OPEN, None, src_bin,(-1, -1), 5)
  src_bin = cv.medianBlur(src_bin, 3)
  
  cnt, labels, stats, centroids = cv.connectedComponentsWithStats(src_bin)

  dst = cv.cvtColor(src2, cv.COLOR_GRAY2BGR)
  arr = []

  for i in range(1, cnt):
    (x, y, w, h, area) = stats[i]

    if area < 8000 or area > 20000:
      continue

    pt1 = (x, y)
    pt2 = (x + w, y + h)

    cv.rectangle(dst, pt1, pt2, (0, 255, 255))

    tmp = src_bin[y:y+h, x:x+w] 
    
    circle_cnt = hough_circles(tmp)
    arr.append(circle_cnt)
  
  # cv.imshow('src_bin',src_bin)
  # cv.imshow('dst', dst)

  # cv.waitKey()
  # cv.destroyAllWindows()

  return arr


for i in range(1,5):
  filename = 'img3_{}.png'.format(i)
  print(filename)
  src = cv.imread(filename, cv.IMREAD_COLOR)
  src2 = color_split(src)
  arr = labeling_stats(src2)

  arr.sort()
  print(arr)