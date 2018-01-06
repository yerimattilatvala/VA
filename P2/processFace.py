import numpy as np
import cv2
import argparse
import imutils
import sys
from utils import *
from PIL import Image, ImageEnhance
#from k import *
#from filtroHomorfico import *

#coger argumentos de entrada
#---------------------------------------------------------------------------#
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagen", required=True)
args = vars(ap.parse_args()) #{'plantilla':'plantilla','imagen': 'img'}
#---------------------------------------------------------------------------#

#CARGAR IMAGEN Y PREPROCESAR
#---------------------------------------------------------------------------#
img = cv2.imread(args['imagen'])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #imagen a escala de grises
equal = cv2.equalizeHist(gray) #ecualizacion de histograma para imagenes oscuras
#---------------------------------------------------------------------------#

#CARGAR CLASIFICADOR HAARCASCADE
#---------------------------------------------------------------------------#
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
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

if face is None: #SI EL CLASIFICADOR NO ENCUENTRA LA CARA -> TERMINAR EJECUCION
    print("No se ha encontrado la cara\n")
    sys.exit()
#---------------------------------------------------------------------------#

#ENCONTRAMOS LOS OJOS
#---------------------------------------------------------------------------#
f,c = face.shape[0],face.shape[1]
roi1 = face[:int(round(f/2)),]
f1,c1 = roi1.shape[0],roi1.shape[1]
roi1 = roi1[int(round(f1/1.75)):,int(round(c1/4.5)):-int(round(c1/4.5))]
part = int(round(roi1.shape[1]/3))
ojoDerecho = roi1[:,:part]
ojoIzquierdo = roi1[:,2*part:]
ojoDerecho_g = cv2.cvtColor(ojoDerecho,cv2.COLOR_BGR2GRAY)
ojoIzquierdo_g = cv2.cvtColor(ojoIzquierdo,cv2.COLOR_BGR2GRAY)
ojoDerecho_g = cv2.equalizeHist(ojoDerecho_g) 
ojoIzquierdo_g = cv2.equalizeHist(ojoIzquierdo_g) 
#ojoDerecho_g = cv2.bilateralFilter(ojoDerecho_g,9,75,75)
#ojoIzquierdo_g = cv2.bilateralFilter(ojoIzquierdo_g,9,75,75)
#ojoDerecho_g = auto_canny(ojoDerecho_g)
#ojoIzquierdo_g = auto_canny(ojoIzquierdo_g)
#detectC(ojoDerecho)
#detectC(ojoIzquierdo)
L1 = hougN(ojoDerecho)
L2 = hougN(ojoIzquierdo)
#hsv(ojoDerecho)
#hsv(ojoIzquierdo)
lookSide = L1 + L2
print (lookSide)
'''lookSide = [] #lista para guardar la distintas direcciones de los ojos que puede pillar(A LO MEJOR SE PUEDEN ELIMINAR AQUELLOS QUE NO CREAMOS UTILES)

lookSide.append(plantilla(ojoDerecho_g))

lookSide.append(plantilla(ojoIzquierdo_g))
print(lookSide)'''
#---------------------------------------------------------------------------#
side = 'undefined'
if not lookSide:
    print("No se ha definido la direccion de la mirada\n")
    sys.exit()
elif ((lookSide.count('Right')) == (len(lookSide))) or ((lookSide.count('Right')) > (lookSide.count('Left'))):
    side = 'Right'
elif ((lookSide.count('Left')) == (len(lookSide))) or ((lookSide.count('Right')) < (lookSide.count('Left'))):
    side = 'Left'
#---------------------------------------------------------------------------#
#--------------------------------VISUALIZAMOS-------------------------------#
#---------------------------------------------------------------------------#
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2
cv2.putText(img ,side, bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
cv2.imshow(side,img)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
#---------------------------------------------------------------------------#