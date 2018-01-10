import cv2
import numpy as np
from matplotlib import pyplot as plt
from filtroHomorfico import *
import skimage
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from skimage import exposure
from utils import *


def houg2(image):
    width = 150
    h,w = image.shape[0],image.shape[1]
    r = width / float(w)
    dim = (width, int(h * r))

    image = aumentarContraste(image)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    image = cv2.fastNlMeansDenoisingColored(image,None,11,11,7,21)
    cv2.imshow('',image)

    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image =  cv2.GaussianBlur(image,(11,11),9)
    image = cv2.equalizeHist(image)
    edges = canny(image, sigma=2, low_threshold=15, high_threshold=50)
    # Detect two radii
    hough_radii = np.arange(20, 50, 1)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 5 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                            total_num_peaks=3)
    valores = []
    radio1 = []
    for center_y, center_x, radius in zip(cy, cx, radii):
        valores.append(image[center_y,center_x])
        radio1.append(radius)
    #print(valores)
    minimo = np.amin(valores)
    maxr = np.max(radio1)
    #print(minimo)
    mitad = int(round(image.shape[1]/2))
    '''cv2.imshow("",image[:,:mitad])
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    cv2.imshow("",image[:,mitad:])
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()'''
    #fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    image1 = color.gray2rgb(image)
    side = []

    y = []
    x = []
    radio = []
    for center_y, center_x, radius in zip(cy, cx, radii):
        value = image[center_y,center_x]
        #if (center_y > part) and (center_y < (2*part)) and (value == minimo):
        if (radius == maxr):
            y.append(center_y)
            x.append(center_x)
            radio.append(radius)
    x = int(np.mean(x))
    y = int(np.mean(y))
    radio = int(np.mean(radio))
    cv2.circle(image1,(x,y),radio,(0,255,0),2)
    cv2.imshow('',image1)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()

def hougN(image):
    width = 150
    h,w = image.shape[0],image.shape[1]
    r = width / float(w)
    dim = (width, int(h * r))
    
    image = aumentarContraste(image)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    image = cv2.fastNlMeansDenoisingColored(image,None,11,11,7,21)

    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image =  cv2.GaussianBlur(image,(7,7),7.6)
    image = cv2.equalizeHist(image)
    edges = canny(image, sigma=2, low_threshold=15, high_threshold=50)
    # Detect two radii
    hough_radii = np.arange(20, 30, 1)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 5 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                            total_num_peaks=3)
    #print(accums,cx,cy,radii)
    part = int(round(image.shape[0]/3))   
    '''cv2.imshow("",image[part:(2*part),:])
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()'''
    # Draw them
    valores = []
    for center_y, center_x, radius in zip(cy, cx, radii):
        valores.append(image[center_y,center_x])
    #print(valores)
    minimo = np.amin(valores)
    #print(minimo)
    mitad = int(round(image.shape[1]/2))
    '''cv2.imshow("",image[:,:mitad])
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    cv2.imshow("",image[:,mitad:])
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()'''
    #fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    image1 = color.gray2rgb(image)
    side = []

    y = []
    x = []
    radio = []
    for center_y, center_x, radius in zip(cy, cx, radii):
        value = image[center_y,center_x]
        #if (center_y > part) and (center_y < (2*part)) and (value == minimo):
        if (value == minimo):
            y.append(center_y)
            x.append(center_x)
            radio.append(radius)
    
    if not x or not y or not radio:
        print("Ojo no definido\n")
        return []
    x = int(np.mean(x))
    y = int(np.mean(y))
    radio = int(np.mean(radio))
    #print(x,y,radio)
    cv2.circle(image,(x,y),radio,(0,255,0),2)
    image_aux = image[y-radio:y+radio,x-radio:x+radio] #se usa y para detectar
    #image = kMeans(image,2)
    if (x-(radio*2)) > 0:
        dch = image[int(round(y-radio/2)):int(round(y+radio/2)),x-(radio*2):x-radio]
        izq = image[int(round(y-radio/2)):int(round(y+radio/2)),x+radio:x+(radio*2)]
    else:
        print(x,radio,x-(radio+int(round(radio/5))),x-radio)
        dch = image[int(round(y-radio/2)):int(round(y+radio/2)),x-(radio+int(round(radio/5))):x-radio]
        izq = image[int(round(y-radio/2)):int(round(y+radio/2)),x+radio:x+radio+int(round(radio/5))]
    #print(dch.shape,izq.shape)
    dch = kMeans(dch,2)
    izq = kMeans(izq,2)
    #image_aux = kMeans(image_aux,2)
    '''cv2.imshow("",image_aux)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    cv2.imshow("",dch)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    cv2.imshow("",izq)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()'''
    
    print('PARTE DERECHA BLANCO : ',np.sum(dch == np.max(dch)),'NEGRO :',np.sum(dch == np.min(dch)))
    print('PARTE IZQUIERDA BLANCO : ',np.sum(izq == np.max(izq)),'NEGRO :',np.sum(izq == np.min(izq)))
    circy, circx = circle_perimeter(y,x, radio)
    image1[circy, circx] = (220, 20, 20)
    if (x < mitad):
        side.append('Right')
    elif (x > mitad):
        side.append('Left')
    else:
        side.append('Center')

    #ax.imshow(image1, cmap=plt.cm.gray)
    #plt.show()
    return side, np.sum(dch == np.max(dch)),np.sum(izq == np.max(izq))