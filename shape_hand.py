# -*- coding: utf-8 -*-
import cv2
import numpy as np

img = cv2.imread('hand-learn/img_five010.png')
img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img_gray, 127, 255,0)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]

#img = cv2.drawContours(img, [cnt], 0, (0,255,0), 3)

img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#重心を求めて表示
#mu = cv2.moments(img_gray, False)
#mx,my= int(mu["m10"]/mu["m00"]) , int(mu["m01"]/mu["m00"])
#cv2.circle(img, (mx,my), 4, 100, 2, 4)

#hand_point = []

#for p in cnt:
#    point = tuple(p[0])
#    if my > point[1]:
#        hand_point.append(point)

#for i in range(len(hand_point) - 1):
#    start = hand_point[i]
#    end = hand_point[i+1]
#    cv2.line(img,start,end,[0,255,0],2)

#膨張
kernel = np.ones((5,5),np.uint8)
img = cv2.dilate(img,kernel,iterations = 1)


cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
