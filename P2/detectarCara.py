import cv2
import argparse
import sys
from funciones import *

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
scale_factor=1.3
min_neighbors=5
#DETECTAR CARA
faces = face_cascade.detectMultiScale(equal, scale_factor, min_neighbors)    #usa la imagen en escala de grises para detectar mejor

#---------------------------------------------------------------------------#

#----------------------------BUSCAMOS LA CARA---------------------------#

face = None
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
    x1 = x
    y1 = y
    x2 = x+w
    y2 = y+h
    face = img[y:y+h, x:x+w] 

if face is None: #SI EL CLASIFICADOR NO ENCUENTRA LA CARA -> TERMINAR EJECUCION
    print("No se ha encontrado la cara\n")
    sys.exit()

#---------------------------------------------------------------------------#

#----------------------------BUSCAMOS REGIONES---------------------------#

x,y = punto_medio(img, x1, y1, x2, y2)  # Hallamos el punto medio de la cara.
division_vertical = dividir_verticalmente(img, x1, y1, x2, y2)
division_horizontal = dividir_horizontalmente(img, y, x1, y1, x2, division_vertical)
ojod = ojo_derecho(img, x, y, x1, y1, x2, y2, division_horizontal, division_vertical)
ojoi = ojo_izquierdo(img, x, y, x1, y1, x2, y2, division_horizontal, division_vertical)
pd, cxd, cyd, xd, yd, rde, rd = detectar_pupila(ojod)
pi, cxi, cyi, xi, yi, rie, ri = detectar_pupila(ojoi)
radio_medio = int(round((rd + ri)/2))
centroxd = x1 + division_horizontal + cxd
centroyd = y1 + division_vertical + cyd
centroxi = x1 + (3 * division_horizontal) + cxi-20
centroyi = y1 + division_vertical + cyi
puntos_ojos = recta_ojos(img, centroxd, centroyd, centroxi, centroyi, x, rd, ri)
puntosd = detectar_esclerotica(ojod, xd, yd, rde)
puntosi = detectar_esclerotica(ojoi, xi, yi, rie)
puntos_normalizados = []
for item in puntosd:
    puntoxd = x1 + division_horizontal + item[0]
    puntoyd = y1 + division_vertical + item[1]
    puntos_normalizados.append((puntoxd,puntoyd))

for item1 in puntosi:
    puntoxi = x1 + (3 * division_horizontal) + item1[0] - 20 
    puntoyi = y1 + division_vertical + item1[1] 
    puntos_normalizados.append((puntoxi,puntoyi))

ptos_ojo_d_d, ptos_ojo_d_i, ptos_ojo_i_d, ptos_ojo_i_i = limite_esclerotica(img, puntos_normalizados, puntos_ojos, centroxd, centroxi, radio_medio)
'''print(ptos_ojo_d_d)
print(ptos_ojo_d_i)
print(ptos_ojo_i_d)
print(ptos_ojo_i_i)'''

distancia_ptos_d_a_ojod = distancia_a_pupila(ptos_ojo_d_d, centroxd, centroyd)
distancia_ptos_i_a_ojod = distancia_a_pupila(ptos_ojo_d_i, centroxd, centroyd)
distancia_ptos_d_a_ojoi = distancia_a_pupila(ptos_ojo_i_d, centroxi, centroyi)
distancia_ptos_i_a_ojoi = distancia_a_pupila(ptos_ojo_i_i, centroxi, centroyi)
'''print(distancia_ptos_d_a_ojod)
print(distancia_ptos_i_a_ojod)
print(distancia_ptos_d_a_ojoi)
print(distancia_ptos_i_a_ojoi)'''
media_ojod_d = media_distancias(distancia_ptos_d_a_ojod)
media_ojod_i = media_distancias(distancia_ptos_i_a_ojod)
media_ojoi_d = media_distancias(distancia_ptos_d_a_ojoi)
media_ojoi_i = media_distancias(distancia_ptos_i_a_ojoi)
print(media_ojod_d)
print(media_ojod_i)
print(media_ojoi_d)
print(media_ojoi_i)
for item in puntos_ojos:
    cv2.circle(img,item,1,(255,0,0),2)
cv2.circle(img,(centroxd,centroyd),rd,(255,0,0),1)
cv2.circle(img,(centroxi,centroyi),ri,(255,0,0),1)
lineThickness = 1
#cv2.line(img, (centroxd,centroyd), (centroxi,centroyi),(255, 0, 0), lineThickness)
#cv2.line(img, (centroxd-2*rd,centroyd), (centroxd + rd*2,centroyd),(0, 100, 255), lineThickness)
#---------------------------------------------------------------------------#

#--------------------------------VISUALIZAMOS-------------------------------#
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2
cv2.putText(img ,determinar_lado(media_ojod_d, media_ojod_i, media_ojoi_d, media_ojoi_i), bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
cv2.imshow("",img)
cv2.imshow("pd",pd)
cv2.imshow("pi",pi)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
'''plt.axis("off")
plt.imshow(img)
plt.show()'''

#---------------------------------------------------------------------------#

