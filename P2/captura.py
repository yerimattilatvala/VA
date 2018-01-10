import numpy as np
import cv2
import argparse
import imutils
import sys
from utils import *
from utils2 import *
from PIL import Image, ImageEnhance

#---------------------------------------------------------------------------#
img = cv2.imread('4.Eyes_and_gaze\Derecha10.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #imagen a escala de grises
equal = cv2.equalizeHist(gray) #ecualizacion de histograma para imagenes oscuras
#---------------------------------------------------------------------------#

#CARGAR CLASIFICADOR HAARCASCADE
#---------------------------------------------------------------------------#
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
scale_factor=1.3
min_neighbors=5
#DETECTAR CARA
faces = face_cascade.detectMultiScale(equal, scale_factor, min_neighbors)    #usa la imagen en escala de grises para detectar mejor
#---------------------------------------------------------------------------#

#----------------------------PILLAMOS REGION CARA---------------------------#
face = None
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    face = img[y:y+h, x:x+w] 
    face_gray = equal[y:y+h, x:x+w] 

eyes = eye_cascade.detectMultiScale(face_gray,scale_factor,min_neighbors)

for (ex,ey,ew,eh) in eyes:
    cv2.rectangle(face,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) #para marcar los ojos en la imagen en color
    eye = face[ey:ey+eh,ex:ex+ew]#aqui seleccionamos el contorno de cada ojo
    part = int(round(eye.shape[0]/3))
    part2 = int(round(eye.shape[1]/6))
    eye = eye[part:part*2,part2:-part2]
    houg2(eye)
    cv2.imshow('',eye)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()

cv2.imshow('',face)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()