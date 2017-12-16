import numpy as np
import cv2

ruta = '4. Eyes and gaze'


def detect(image):
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale_factor=1.3
    min_neighbors=3
    faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)    #usa la imagen en escala de grises para detectar mejor
    look = 'undefined'
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]   #selecciona las regiones de interes dentro de la cara en escala de grises y detectar los ojos
        roi_color = img[y:y+h, x:x+w]   #idem al anterior pero en color para dibujar los rectangulos en color
        eyes = eye_cascade.detectMultiScale(roi_gray,scale_factor,min_neighbors)
        for (ex,ey,ew,eh) in eyes:
            eye = cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) #para marcar los ojos en la imagen en color
            eye = cv2.cvtColor(eye[ey:ey+eh,ex:ex+ew],cv2.COLOR_BGR2GRAY) #aqui seleccionamos el contorno de cada ojo
            #eye = eye[ey:ey+eh,ex:ex+ew]
            leftPartEye = eye[:,:int(round(eye.shape[1]/2))]
            rightPartEye = eye[:,int(round(eye.shape[1]/2)):]
            if ((np.sum(leftPartEye == 255))>(np.sum(rightPartEye == 255))):
                look = 'Right' 
            else:
                look = 'Left'    
    return img,look  

def show(image,side):
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
imagen = '\Derecha8.jpg'
img,side = detect(ruta+imagen)
show(img,side)