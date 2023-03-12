import cv2 as cv

def hough_circles(src):
  blurred = cv.blur(src, (3, 3))
  circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, 1, 10, param1=150, param2=20)
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

def labeling_stats(src):
  _, src_bin = cv.threshold(src, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

  cnt, labels, stats, centroids = cv.connectedComponentsWithStats(src_bin)

  dst = cv.cvtColor(src, cv.COLOR_GRAY2BGR)
  arr = []

  for i in range(1, cnt):
    (x, y, w, h, area) = stats[i]

    if area < 20:
      continue

    pt1 = (x, y)
    pt2 = (x + w, y + h)

    cv.rectangle(dst, pt1, pt2, (0, 255, 255))
    tmp = src[y:y+h, x:x+w]

    circle_cnt = hough_circles(tmp)
    arr.append(circle_cnt)

  # cv.imshow('dst', dst)
  # cv.waitKey()
  # cv.destroyAllWindows()
  
  return arr

for i in range(1, 3):
  filename = 'img2_{}.png'.format(i)
  print(filename)
  src = cv.imread(filename, cv.IMREAD_GRAYSCALE)
  
  arr = labeling_stats(src)
  arr.sort()
  print(arr)
