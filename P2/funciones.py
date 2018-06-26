import cv2
from math import sqrt
import numpy as np
from matplotlib import pyplot as plt
from filtroHomorfico import *
import skimage
from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.feature import corner_harris, corner_subpix, corner_peaks, blob_dog
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from skimage import exposure
from skimage.filters import threshold_otsu, threshold_local, threshold_adaptive, laplace, gaussian
from skimage.morphology import disk
from skimage.filters.rank import median

def punto_mediop(x1, y1, x2, y2):
    x = int((x1+x2)/2)
    y = int((y1+y2)/2)
    return x,y

def aumentarContraste(img):
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2,a,b))  # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    return img2

def detectar_pupila(image):
    
    img = aumentarContraste(image)    #mejoramos contraste imagen en color
    #---------------------------------------------------------------------------#
    width = 150
    h,w = image.shape[0],image.shape[1]     #redimensionamos imagen
    r = width / float(w)
    dim = (width, int(h * r))
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #---------------------------------------------------------------------------#
    
    img = cv2.fastNlMeansDenoisingColored(img,None,11,11,7,21)  #eliminamos ruido

    #---------------------------------------------------------------------------#
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #convertimos a escala de grises
    #
    img = cv2.equalizeHist(img)
    img = median(img,disk(3))
    img =  cv2.GaussianBlur(img,(7,7),7.6)  #suavizamos y mejoramos contraste
    #---------------------------------------------------------------------------#
    edges = canny(img, sigma=2, low_threshold=15, high_threshold=50) #DETECTAR BORDES
    
    hough_radii = np.arange(20, 30, 1)  # Aproximamos radio de la pupilas
    hough_res = hough_circle(edges, hough_radii)    # buscamos la pupila 

    #coordenadas y radios
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,total_num_peaks=3)
        
    valores = []
    for center_y, center_x, radius in zip(cy, cx, radii):
        valores.append(img[center_y,center_x])

    minimo = np.amin(valores)
    image1 = color.gray2rgb(img)
    y = []
    x = []
    radio = []

    for center_y, center_x, radius in zip(cy, cx, radii):
        value = img[center_y,center_x]
        if (value == minimo):
            y.append(center_y)
            x.append(center_x)
            radio.append(radius)
    
    if not x or not y or not radio:
        print("Ojo no definido\n")
        return None

    x = int(np.mean(x))
    y = int(np.mean(y))
    radio = int(np.mean(radio))

    cv2.circle(image1,(x,y),radio,(0,255,0),2)
    
    h1, w1 = image1.shape[0], image1.shape[1]
    ctx = int(round((h*x)/h1))
    cty = int(round((y*w)/w1))
    radio2 = int(round((radio * h)/h1))
    return image1, ctx, cty, x, y, radio, radio2

def filtrar_esquinas(limites,esquinas, x, y, radio):
    #print(x,y,radio)
    mitad_radio = int(round(radio/2))

    fila_inicio = y - radio
    if fila_inicio < 0 :
        fila_inicio = y - int(round(radio/3))

    fila_fin = y + radio
    if fila_fin > limites[0]:
        fila_fin = y + int(round(radio/3))

    columna_inicio = x - radio
    if columna_inicio<0 :
        columna_inicio = x - radio
    
    columna_fin = x + radio
    if columna_fin >=limites[1]:
        columna_fin = x + radio

    puntos = [] 
    for i in range(len(esquinas)):
        if esquinas[i][0]>fila_inicio and esquinas[i][0]<fila_fin:
            if (esquinas[i][1]< columna_inicio and esquinas[i][1] > x - 3 * radio) or (esquinas[i][1] > columna_fin and esquinas[i][1]< x + 3 * radio):
                puntos.append(esquinas[i])
    
    esquinas = np.array(puntos)
    #print(esquinas)
    return esquinas

def detectar_esclerotica(image, coord_pupila_x, coord_pupila_y, radio):
    img = aumentarContraste(image)    #mejoramos contraste imagen en color
    #---------------------------------------------------------------------------#
    width = 150
    h,w = img.shape[0],img.shape[1]     #redimensionamos imagen
    r = width / float(w)
    dim = (width, int(h * r))
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #---------------------------------------------------------------------------#
    #img = cv2.fastNlMeansDenoisingColored(img,None,11,11,7,21)  #eliminamos ruido
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #convertimos a escala de grises
    #img =  cv2.blur(img,(5,5))  #suavizamos y mejoramos contraste
    #img = cv2.equalizeHist(img)
    elemento_estructurado = np.ones((5,5), dtype=np.uint8)
    mediana = median(img,elemento_estructurado)
    substraida = img - mediana
    img = img + substraida
    img = gaussian(img,(3,3),1)
    '''unsharp_strength = 0.9
    blur_size = 8  # Standard deviation in pixels.

    # Convert to float so that negatives don't cause problems
    img1 = skimage.img_as_float(img)
    blurred = gaussian(img1, blur_size)
    highpass = img1 - unsharp_strength * blurred
    sharp = img1 + highpass
    cv2.imshow("img",img1)
    cv2.imshow("sharp",sharp)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    lap = laplace(img,7)
    lap = lap * 255
    lap = lap.astype(np.uint8)
    t = img + lap
    img = t.astype(np.uint8)
    img = gaussian(img,(3,3),1)
    cv2.imshow("img",img
    cv2.imshow("lap",lap)
    cv2.imshow("t",t)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    sharp = sharp * 255
    img = sharp.astype(np.uint8)
    print(type(sharp[0][0]),type(img[0][0]))'''
    
    #---------------------------------------------------------------------------#
    global_thresh = threshold_otsu(img)
    binary_global = img > global_thresh
    block_size = 35
    local_thresh = threshold_local(img, block_size)
    binary_local = img > local_thresh
    binary_local.astype(float)
    binary_global.astype(float)
    #---------------------------------------------------------------------------#
    edges = canny(img, sigma=2, low_threshold=15, high_threshold=50) #DETECTAR BORDES
    edges = edges.astype(float)
    #---------------------------------------------------------------------------#
    coords = corner_peaks(corner_harris(binary_local), min_distance=5)
    #coords1 = corner_peaks(corner_harris(edges), min_distance=5)

    #coords_subpix = corner_subpix(edges, coords, window_size=3)

    '''fig, ax = plt.subplots()
    ax.imshow(binary_local, interpolation='nearest', cmap=plt.cm.gray)
    ax.plot(coords[:, 1], coords[:, 0], '.b', markersize=3)
    #ax.plot(coords1[:, 1], coords1[:, 0], '+r', markersize=5)
    ax.axis((0, 350, 350, 0))
    plt.show()'''

    coords = filtrar_esquinas(img.shape,coords, coord_pupila_x, coord_pupila_y, radio)
    #coords1 = filtrar_esquinas(img.shape,coords1, coord_pupila_x, coord_pupila_y, radio)

    '''fig, ax = plt.subplots()
    ax.imshow(binary_local, interpolation='nearest', cmap=plt.cm.gray)
    ax.plot(coords[:, 1], coords[:, 0], '.b', markersize=3)
    #ax.plot(coords1[:, 1], coords1[:, 0], '+r', markersize=5)
    ax.axis((0, 350, 350, 0))
    plt.show()'''

    h1, w1 = binary_local.shape[0], binary_local.shape[1]
    #print(h,w,h1,w1)
    coord_normalizadas = []
    for x in coords:
        ctx = int(round((h*x[0])/h1))
        cty = int(round((x[1]*w)/w1))
        item = [cty, ctx]
        coord_normalizadas.append(item)
    

    '''for x in coord_normalizadas:
       cv2.circle(image,(x[0],x[1]),1,(0,255,0),2)
    cv2.imshow("img",image)
    cv2.imshow("img1",img)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()'''
    
    return coord_normalizadas

def elementos_por_coordenada(x, lista_coord):
    contador = 0
    for item in lista_coord:
        if item[0][0] == x[0] and item[0][1] == x[1]:
            contador += 1
    
    return x,contador

def filtrar_puntos(lista_coord):
    ptos_con_contador = []
    for pto in lista_coord:
        ptos_con_contador.append(elementos_por_coordenada(pto[0], lista_coord))

    maximo = 0
    pto = None
    for ptos in ptos_con_contador:
        if (ptos[1] > maximo):
            maximo = ptos[1]
            pto = ptos[0]

    lista_filtrada = []

    for ptos in lista_coord:
        if pto == ptos[0]:
            lista_filtrada.append(ptos)

    return lista_filtrada

def limite_esclerotica(image, lista_puntos, puntos_ojos, xd, xi, radio):
    ojo_d_d = []
    ojo_d_i = []
    ojo_i_d = []
    ojo_i_i = []
    for x in puntos_ojos:
        for x1 in lista_puntos:
            if (x[1] < x1[1] and x1[1]< x[1]+ int(round(radio/1.25))) and (x1[0]> x[0] - int(round(radio/1.10)) and x1[0] < x[0] + int(round(radio/1.10))) :
                if x[0] < xd :
                    ojo_d_d.append((x,x1))
                elif x[0] > xi :
                    ojo_i_i.append((x,x1))
                elif x[0] > xd and x[0] < (xd + 4 *radio):
                    ojo_d_i.append((x,x1))
                else:
                    ojo_i_d.append((x,x1))
                #cv2.circle(image,(x1[0],x1[1]),1,(0,255,0),2)

    ojo_d_d = filtrar_puntos(ojo_d_d)
    ojo_d_i = filtrar_puntos(ojo_d_i)
    ojo_i_d = filtrar_puntos(ojo_i_d)
    ojo_i_i = filtrar_puntos(ojo_i_i)
    puntos = []
    for item in ojo_d_d:
        cv2.circle(image,(item[1]),1,(0,255,0),2)
    for item in ojo_d_i:
        cv2.circle(image,(item[1]),1,(0, 0,255),2)
    for item in ojo_i_d:
        cv2.circle(image,(item[1]),1,(255,0,255),2)
    for item in ojo_i_i:
        cv2.circle(image,(item[1]),1,(255,255,0),2)

    return ojo_d_d, ojo_d_i, ojo_i_d, ojo_i_i

def punto_medio(img, x1, y1, x2, y2):
    x = int((x1+x2)/2)
    y = int((y1+y2)/2)
    lineThickness = 1
    cv2.line(img, (x, y1), (x, y2), (0,255,0), lineThickness)
    cv2.line(img, (x1, y), (x2, y), (0,255,0), lineThickness)
    return x,y

def dividir_verticalmente(img, x1, y1, x2, y2):
    division_vertical = int(round((y2-y1)/3))
    '''lineThickness = 1
    for i in range(0,4):
        cv2.line(img, (x1, y1+(i*division_vertical)), (x2, y1+(i*division_vertical)), (0, 0, 255), lineThickness)'''
    
    return division_vertical

def dividir_horizontalmente(img, y, x1, y1, x2, division_vertical):
    division_horizontal = int(round((x2-x1)/5))
    lineThickness = 1
    #for i in range(0,6):
        #cv2.line(img, (x1+(i*division_horizontal), y1+division_vertical), (x1+(i*division_horizontal), y), (255,255,0), lineThickness)

    return division_horizontal

def ojo_derecho(img, x, y, x1, y1, x2, y2, d_horizontal, d_vertical):
    ojo_derecho = img[ y1 + d_vertical : y, x1 + d_horizontal : x1 + (2 * d_horizontal)+20 ]

    return ojo_derecho

def ojo_izquierdo(img, x, y, x1, y1, x2, y2, d_horizontal, d_vertical):
    ojo_izquierdo = img[ y1 + d_vertical : y, x1 + (3 * d_horizontal) -20 : x1 + (4 * d_horizontal) ]

    return ojo_izquierdo

def puntos_proximos(x, lista):
    aux = x
    for item in lista:
        if x[0] != item[0]:
            if np.abs(item[0] -x[0]) < 10 or np.abs(x[0] - item[0]) < 10  :
                aux = punto_mediop(aux[0], aux[1], item[0], item[1])
                #print(x,item,aux)
    return aux

def elemento_en_lista(x,lista):
    dentro = False
    for item in lista:
        if item[0] == x[0] and x[1] == item[1]:
            dentro = True
            break
    return dentro

def filtrar_puntos_ojos(puntos, centrox,rd, ri):
    lista = []
    for x in puntos:
        if np.abs(x[0] - centrox) > rd*2 and np.abs(x[0] - centrox) > ri*2:
            lista.append(x)
    
    lista_filtrada = []
    for x in lista:
        punto = puntos_proximos(x, lista)
        if elemento_en_lista(punto, lista_filtrada) == False: 
            #print('FILTRADO-> ',punto)
            lista_filtrada.append(punto)
    #print('--------------------------------')
    lista = []
    for x in lista_filtrada:
        punto = puntos_proximos(x, lista_filtrada)
        if elemento_en_lista(punto, lista) == False:
            #print('FILTRADO-> ',punto)
            lista.append(punto)

    return lista


def recta_ojos(img,x1, y1, x2, y2, centrox, rd, ri):
    puntos_recta = []
    m1 = x2 - x1
    m2 = y2 -y1
    a = x1 * m2
    b = y1 * m1
    for x in range(x1 -40,x2 + 40):
        for y in range(img.shape[0]):
            z = (x * m2) - a
            w = (y * m1) - b
            if (np.abs(z - w) <20 or np.abs(w - z)<20):
                item = (x,y)
                puntos_recta.append(item)
                #cv2.circle(img,(x,y),1,(0,255,0),1)

    puntos_recta = filtrar_puntos_ojos(puntos_recta, centrox ,rd, ri)
    return puntos_recta
    
def distancia_entre_dos_puntos(x1, y1, x2, y2):
    a = x2 - x1
    b = y2 - y1
    d = sqrt((a**2) + (b**2))

    return d

def distancia_a_pupila(ptos, xp, yp):
    lista_distancias = []

    for pto in ptos:
        lista_distancias.append(distancia_entre_dos_puntos(pto[1][0], pto[1][1], xp, yp))

    return lista_distancias

def media_distancias(lista_puntos):
    media = 1000
    if len(lista_puntos)>0:
        c = len(lista_puntos)
        a = np.sum(lista_puntos)
        media = a / c

    return media

def determinar_lado(media_ojod_d, media_ojod_i, media_ojoi_d, media_ojoi_i):
    ojod = 0
    ojoi = 0
    ojod_min_valor = min(media_ojod_d, media_ojod_i)
    ojoi_min_valor = min(media_ojoi_d, media_ojoi_i)
    
    if np.abs(media_ojod_d - media_ojod_i) > 6:
        if ojod_min_valor == media_ojod_d:
            ojod = 1
        else:
            ojod = 2
    if np.abs(media_ojoi_d - media_ojoi_i) < 6:
        if ojod != 0:
            ojoi = ojod
    else:
        if ojoi_min_valor == media_ojoi_d:
            ojoi = 1
        else:
            ojoi = 2
        if ojod == 0:
            ojod = ojoi

    lado = ""
    if ojod == 1 and ojoi == 1:
        lado += "Derecha"
    elif ojod == 2 and ojoi == 2:
        lado += "Izquierda"
    else:
        lado += "Indefinido"

    return lado