import numpy as np
import cv2
import argparse
import imutils

#coger argumentos de entrada
#---------------------------------------------------------------------------#
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plantilla", required=True, help="Path para la plantilla")
ap.add_argument("-i", "--imagen", required=True,
	help="Path para imagen")
args = vars(ap.parse_args()) #{'plantilla':'plantilla','imagen': 'img'}
#---------------------------------------------------------------------------#

#CARGAR IMAGEN Y PREPROCESAR
#---------------------------------------------------------------------------#
img = cv2.imread(args['imagen'])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #imagen a escala de grises
equal = cv2.equalizeHist(gray) #ecualizacion de histograma para imagenes oscuras
#---------------------------------------------------------------------------#

#CARGAR PLANTILLA
#---------------------------------------------------------------------------#
plantilla = cv2.imread(args['plantilla'])
plantilla_gray = cv2.cvtColor(plantilla, cv2.COLOR_BGR2GRAY) #imagen a escala de grises
plantilla_gray = cv2.Canny(plantilla_gray, 50, 200)
#---------------------------------------------------------------------------#

#CARGAR CLASIFICADOR HAARCASCADE
#---------------------------------------------------------------------------#
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
scale_factor=1.3
min_neighbors=3
#DETECTAR CARA
faces = face_cascade.detectMultiScale(equal, scale_factor, min_neighbors)    #usa la imagen en escala de grises para detectar mejor
#---------------------------------------------------------------------------#

for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

#--------------------------------VISUALIZAMOS-------------------------------#
#---------------------------------------------------------------------------#
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2
cv2.putText(img,"", bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
cv2.imshow("",img)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
#---------------------------------------------------------------------------#