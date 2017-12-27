import numpy as np
import cv2
import argparse
import imutils
import sys
from utils import *
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
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
scale_factor=1.3
min_neighbors=3
#DETECTAR CARA
faces = face_cascade.detectMultiScale(equal, scale_factor, min_neighbors)    #usa la imagen en escala de grises para detectar mejor
#---------------------------------------------------------------------------#

#----------------------------PILLAMOS REGION CARA---------------------------#
roi = None
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi = img[y:y+h, x:x+w] 

if roi is None: #SI EL CLASIFICADOR NO ENCUENTRA LA CARA -> TERMINAR EJECUCION
    print("No se ha encontrado la cara\n")
    sys.exit()
#---------------------------------------------------------------------------#

#---------------------------------------------------------------------------#
f,c = roi.shape[0],roi.shape[1]
roi1 = roi[:int(round(f/2)),]
f1,c1 = roi1.shape[0],roi1.shape[1]
roi2 = roi1[int(round(f1/2)):,]
roi2_g = cv2.cvtColor(roi2,cv2.COLOR_BGR2GRAY)
roi2_g = cv2.equalizeHist(roi2_g)
#---------------------------------------------------------------------------#

eyes = eye_cascade.detectMultiScale(roi2_g,scale_factor,min_neighbors) #usamos clasificador
eyes_list = []
for (ex,ey,ew,eh) in eyes:
    cv2.rectangle(roi2,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) #para marcar los ojos en la imagen en color
    eye = roi2[ey+(int(eh/3)):ey+eh-(int(eh/3)),ex:ex+ew] #selecciono la zona mas proxima a los ojos
    eye_g = cv2.cvtColor(eye,cv2.COLOR_BGR2GRAY)
    eye_g = cv2.equalizeHist(eye_g)
    eye1 = kMeans(eye_g,3) #aplico el algoritmo de division de regiones
    eyes_list.append(eye1)
    # PRUEBAS INFRUCTUOSAS(DE MOMENTO)
    '''cv2.imshow("sad",eye1)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    cv2.imshow("sasdaad",eye)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    eye_g = cv2.cvtColor(eye,cv2.COLOR_BGR2GRAY)
    eye1 = cv2.adaptiveThreshold(eye_g,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,25,2)
    cv2.imshow("sasdaad",eye1)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    eye = blob(eye1) 
    if eye is not None:
        cv2.imshow("sad",eye)
        k = cv2.waitKey(0)
        if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()'''

if (len(eyes_list)) == 0: #SI EL CLASIFICADOR NO ENCUENTRA LA CARA -> TERMINAR EJECUCION
    print("No se ha encontrado los ojos\n")
    sys.exit()

lookSide = [] #lista para guardar la distintas direcciones de los ojos que puede pillar(A LO MEJOR SE PUEDEN ELIMINAR AQUELLOS QUE NO CREAMOS UTILES)
for i in eyes_list:
    part = int(round(i.shape[1]/3)) # SE USA PARA DIVIDIR OJO EN 3 PARTES
    roi1_eye = i[:,:part]
    roi2_eye = i[:,part:2*part]
    roi3_eye = i[:,2*part:]
    part1 = np.sum(roi1_eye == np.min(i)) #MIRAMOS PORCENTAJE DE NEGRO
    part2 = np.sum(roi2_eye == np.min(i))
    part3 = np.sum(roi3_eye == np.min(i))
    part1w = np.sum(roi1_eye == np.max(i)) #MIRAMOS PORCENTAJE DE BLANCO
    part2w = np.sum(roi2_eye == np.max(i))
    part3w = np.sum(roi3_eye == np.max(i))
    print(part1,part2,part3)
    print(part1w,part2w,part3w)
    ''''if ((part1 > part3) and (part2>part3)): #seguro se puede mejorar
    #if ((part1 > part3) or (part2>part3)) and ((part1>part2)or(part2>part1)):
    #if (part1 +part2) > part3:
        lookSide.append('Right')
    #elif (part2+part3)>part1:
    #elif ((part2 > part1) or (part3>part1)):
    else:
        lookSide.append('Left')'''
    if part3 > part1:
        lookSide.append('Left')
    else:
        lookSide.append('Right')
    
    #SI DESCOMENTAS LO SIGUIENTE PUEDES MIRAR SI LOS VA DIVIDIENDO MAS O MENOS BIEN
    cv2.imshow("roi1",roi1_eye)
    cv2.imshow("roi2",roi2_eye)
    cv2.imshow("roi3",roi3_eye)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    print(lookSide)

#---------------------------OBTENER DIRECCION---------------------------------#
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