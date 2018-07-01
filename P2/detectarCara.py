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
radio_medioE = int(round((rde + rie)/2))
centroxd = x1 + division_horizontal + cxd
centroyd = y1 + division_vertical + cyd
centroxi = x1 + (3 * division_horizontal) + cxi - 20
centroyi = y1 + division_vertical + cyi
puntos_ojos = recta_ojos(img, centroxd, centroyd, centroxi, centroyi, x, rd, ri)
puntosd = filtrado_esclerotica(ojod, xd, yd, rde)
puntosi = filtrado_esclerotica(ojoi, xi, yi, rie)
puntos_normalizados_d = []
puntos_normalizados_i = []
for item in puntosd:
    puntoxd = x1 + division_horizontal + item[0][0]
    puntoyd = y1 + division_vertical + item[0][1]
    cv2.circle(img, (puntoxd, puntoyd),1,(255, 255, 0), 2)
    puntos_normalizados_d.append(((puntoxd,puntoyd), item[1],item[2], item[3]))
#puntos_normalizadosd = filtrado_final(puntos_normalizadosd, centroxd, centroyd)
print(puntos_normalizados_d)
print('------------')
for item1 in puntosi:
    #print(item1)
    puntoxi = x1 + (3 * division_horizontal) + item1[0][0] - 20 
    puntoyi = y1 + division_vertical + item1[0][1] 
    cv2.circle(img, (puntoxi, puntoyi),1,(0, 255, 255), 2)
    puntos_normalizados_i.append(((puntoxi,puntoyi), item1[1], item1[2], item1[3]))
print(puntos_normalizados_i)
text = ""
t1 = determinar_mirada(puntos_normalizados_d,centroxd)
if t1 < 0:
    text += "Dcha" 
else : 
    text += "Izda"
text += " // "
t2 = determinar_mirada(puntos_normalizados_i,centroxi)
if t2 < 0:
    text += "Dcha"
else : 
    text += "Izda" 
text += " // "
t = t1 + t2
if t < 0:
    text += "Dcha"
else : 
    text += "Izda"

cv2.circle(img,(centroxd,centroyd),rd,(255,0,0),1)
cv2.circle(img,(centroxi,centroyi),ri,(255,0,0),1)
cv2.line(img, (centroxd,centroyd), (centroxi,centroyi),(255, 0, 0), 1)
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
cv2.putText(img ,text, bottomLeftCornerOfText, font, fontScale,fontColor,lineType)
cv2.imshow("",img)
#cv2.imshow("pd",pd)
#cv2.imshow("pi",pi)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
'''plt.axis("off")
plt.imshow(img)
plt.show()'''

#---------------------------------------------------------------------------#

