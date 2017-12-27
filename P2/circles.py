import cv2
import numpy as np

img = cv2.imread('4.Eyes_and_gaze\Derecha10.jpg',0)
img = cv2.equalizeHist(img) #ecualizacion de histograma para imagenes oscuras
img = cv2.medianBlur(img,5)
img = cv2.Canny(img, 50, 200)
cv2.imshow('',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,2,500,
                            param1=100,param2=100,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('detected circles',cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()