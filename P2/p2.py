import numpy as np
import cv2
from k import *
ruta = '4. Eyes and gaze'


def faceDetector(image):
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    img = cv2.imread(image)
    #------------------------------------------------#
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #imagen a escala de grises
    equal = cv2.equalizeHist(gray) #ecualizacion de histograma para imagenes oscuras
    #------------------------------------------------#
    scale_factor=1.3
    min_neighbors=3
    faces = face_cascade.detectMultiScale(equal, scale_factor, min_neighbors)    #usa la imagen en escala de grises para detectar mejor
    look = 'undefined'
    listaLook = []
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = equal[y:y+h,x:x+w]
        #roi_gray = kMeans(roi_gray,8)
        roi_color = img[y:y+h, x:x+w]   #idem al anterior pero en color para dibujar los rectangulos en color
        eyes = eye_cascade.detectMultiScale(roi_gray,scale_factor,min_neighbors)
        for (ex,ey,ew,eh) in eyes:
            eye = cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) #para marcar los ojos en la imagen en color
            eye =  cv2.cvtColor(eye[ey:ey+eh,ex:ex+ew], cv2.COLOR_BGR2GRAY) #aqui seleccionamos el contorno de cada ojo
            #eye = eye[ey:ey+eh,ex:ex+ew]
            eye = kMeans(eye,8)
            print(np.max(eye),np.min(eye))
            leftPartEye = eye[:,:int(round(eye.shape[1]/2))]
            rightPartEye = eye[:,int(round(eye.shape[1]/2)):]
            show(leftPartEye,listaLook)
            show(rightPartEye,listaLook)
            print(look,listaLook)
            if ((np.sum(leftPartEye == np.min(eye) ))>(np.sum(rightPartEye ==  np.min(eye)))):
                look = 'Right' 
            else:
                look = 'Left' 
            print(look,listaLook)
            listaLook.append(look)
        #----------------------------------------------------#
        #----------------------------------------------------#
        print(listaLook)
    return img,listaLook


def show(image,side):
    if (side.count('undefined'))==1:
        side = 'undefined'
    elif (side.count('Right')) == (len(side)):
        side = 'Right'
    elif (side.count('Left')) == (len(side)):
        side = 'Left'
    else:
        side = 'undefined' 
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10,500)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    cv2.putText(image,side, bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
    cv2.imshow(side,image)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
imagen = '\Derecha2.jpg'
face,look = faceDetector(ruta+imagen)
show(face,look)