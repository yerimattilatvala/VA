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



def kMeans(img,k):
    Z = img.flatten()

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = k
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2

def hsv(imagen):
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
 
    '''#Rango de colores detectados:
    #Verdes:
    verde_bajos = np.array([49,50,50])
    verde_altos = np.array([107, 255, 255])
    #Azules:
    azul_bajos = np.array([100,65,75], dtype=np.uint8)
    azul_altos = np.array([130, 255, 255], dtype=np.uint8)
    #Rojos:
    rojo_bajos1 = np.array([0,65,75], dtype=np.uint8)
    rojo_altos1 = np.array([12, 255, 255], dtype=np.uint8)
    rojo_bajos2 = np.array([240,65,75], dtype=np.uint8)
    rojo_altos2 = np.array([256, 255, 255], dtype=np.uint8)
    
    
    
    #Crear las mascaras
    mascara_verde = cv2.inRange(hsv, verde_bajos, verde_altos)
    mascara_rojo1 = cv2.inRange(hsv, rojo_bajos1, rojo_altos1)
    mascara_rojo2 = cv2.inRange(hsv, rojo_bajos2, rojo_altos2)
    mascara_azul = cv2.inRange(hsv, azul_bajos, azul_altos)
    
    #Juntar todas las mascaras
    mask = cv2.add(mascara_rojo1, mascara_rojo2)
    mask = cv2.add(mask, mascara_verde)
    mask = cv2.add(mask, mascara_azul)
    '''
    blanco_bajo = np.array([0,0,100], dtype=np.uint8)
    blanco_alto = np.array([12, 1, 1], dtype=np.uint8)
    mask = cv2.inRange(hsv,blanco_bajo,blanco_alto)
    #Mostrar la mascara final y la imagen
    cv2.imshow('Finale', mask)
    cv2.imshow('Imagen', imagen)
    
    #Salir con ESC
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()

def plantilla(img):
    #img = np.mean(img, axis=2)
    #img =  cv2.GaussianBlur(img,(5,5),0)
    (M,N) = img.shape[0],img.shape[1]
    part = int(round(img.shape[1]/3))
    mm = M-30
    nn = N -30
    template = np.zeros([mm,nn])
    
    ## Create template ##

    #darkest inner circle (pupil)
    (rr,cc) = skimage.draw.circle(mm/2,nn/2,4.5, shape=template.shape)
    template[rr,cc]=-2
    #iris (circle surrounding pupil)
    (rr,cc) = skimage.draw.circle(mm/2,nn/2,8, shape=template.shape)
    template[rr,cc] = -1
    #Optional - pupil reflective spot (if centered)
    (rr,cc) = skimage.draw.circle(mm/2,nn/2,1.5, shape=template.shape)
    template[rr,cc] = 1
    #plt.imshow(template)
    normccf = skimage.feature.match_template(img, template,pad_input=True)

    #center pixel
    (i,j) = np.unravel_index( np.argmax(normccf), normccf.shape)
    print(i,j)
    mitad = int(round(img.shape[1]/2))
    plt.imshow(img)
    plt.plot(j,i,'r*')
    plt.show()
    if j < mitad :
        lado = 'Right'
    else:
        lado = 'Left'
    return lado

def other2(img_rgb):
    #img_rgb = cv.imread('mario.png')
    (M,N) = img_rgb.shape[0],img_rgb.shape[1]
    mm = M-20
    nn = N-20
    template = np.zeros([mm,nn])
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    #template = cv.imread('mario_coin.png',0)
    w, h = template.shape[0],template.shape[1]
    res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
    #cv.imwrite('res.png',img_rgb)
    cv2.imshow("",img)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()

def elip(image):
    image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges = canny(image_gray, sigma=2.0,low_threshold=0.55, high_threshold=0.8)
    # Perform a Hough Transform
    # The accuracy corresponds to the bin size of a major axis.
    # The value is chosen in order to get a single high accumulator.
    # The threshold eliminates low accumulators
    result = hough_ellipse(edges, accuracy=20, threshold=250,
                        min_size=100, max_size=120)
    result.sort(order='accumulator')

    # Estimated parameters for the ellipse
    best = list(result[-1])
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]

    # Draw the ellipse on the original image
    cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
    image_rgb[cy, cx] = (0, 0, 255)
    # Draw the edge (white) and the resulting ellipse (red)
    edges = color.gray2rgb(img_as_ubyte(edges))
    edges[cy, cx] = (250, 0, 0)

    fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4), sharex=True,
                                    sharey=True,
                                    subplot_kw={'adjustable':'box-forced'})

    ax1.set_title('Original picture')
    ax1.imshow(image_rgb)

    ax2.set_title('Edge (white) and result (red)')
    ax2.imshow(edges)


def hougN(image):
    width = 150
    h,w = image.shape[0],image.shape[1]
    r = width / float(w)
    dim = (width, int(h * r))
    
    image = aumentarContraste(image)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    image = cv2.fastNlMeansDenoisingColored(image,None,11,11,7,21)

    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #image = cv2.equalizeHist(image)
    #v_min, v_max = np.percentile(image, (0.2, 99.8))
    #image = exposure.rescale_intensity(image, in_range=(v_min, v_max))
    '''ret,image = cv2.threshold(image,20,255,cv2.THRESH_BINARY)
    cv2.imshow("",image)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()'''
    image =  cv2.GaussianBlur(image,(7,7),7.6)
    image = cv2.equalizeHist(image)
    edges = canny(image, sigma=2, low_threshold=15, high_threshold=50)
    #image2 = kMeans(image,4)
    # Detect two radii
    hough_radii = np.arange(20, 30, 1)
    hough_res = hough_circle(edges, hough_radii)

    # Select the most prominent 5 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,
                                            total_num_peaks=3)
    #print(accums,cx,cy,radii)
    
    '''for center_y, center_x, radius in zip(cy, cx, radii):
        value = image[cx,cy]
        if value <= 30:'''
    part = int(round(image.shape[0]/3))   
    cv2.imshow("",image[part:(2*part),:])
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    # Draw them
    valores = []
    for center_y, center_x, radius in zip(cy, cx, radii):
        valores.append(image[center_y,center_x])
    #print(valores)
    minimo = np.amin(valores)
    #print(minimo)
    mitad = int(round(image.shape[1]/2))
    cv2.imshow("",image[:,:mitad])
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    cv2.imshow("",image[:,mitad:])
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
    image1 = color.gray2rgb(image)
    side = []
    for center_y, center_x, radius in zip(cy, cx, radii):

        value = image[center_y,center_x]
        if (center_y > part) and (center_y < (2*part)) and (value == minimo):
            if (center_x < mitad):
                side.append('Right')
            elif (center_x > mitad):
                side.append('Left')
            else:
                side.append('Center')
            #cv2.rectangle(image1,(center_x,center_y),(center_x,center_y),(0,255,0),2)
            #cv2.rectangle(image1,(center_x,center_y),(center_x,center_y),(255,0,0),2)
            #print(center_x,center_y)
            circy, circx = circle_perimeter(center_y, center_x, radius)
            image1[circy, circx] = (220, 20, 20)

    ax.imshow(image1, cmap=plt.cm.gray)
    plt.show()
    return side


def draw_circles(img, circles):
    # img = cv2.imread(img,0)
    #cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    for i in circles[0,:]:
    # draw the outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),1,(0,0,255),3)
        cv2.putText(img,str(i[0])+str(',')+str(i[1]), (i[0],i[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255)
    return img

def detect_circles(image_path,img):
    #gray = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    gray_blur = cv2.medianBlur(image_path, 13)  # Remove noise before laplacian
    gray_lap = cv2.Laplacian(gray_blur, cv2.CV_8UC1, ksize=5)
    dilate_lap = cv2.dilate(gray_lap, (3, 3))  # Fill in gaps from blurring. This helps to detect circles with broken edges.
    # Furture remove noise introduced by laplacian. This removes false pos in space between the two groups of circles.
    lap_blur = cv2.bilateralFilter(dilate_lap, 5, 9, 9)
    # Fix the resolution to 16. This helps it find more circles. Also, set distance between circles to 55 by measuring dist in image.
    # Minimum radius and max radius are also set by examining the image.
    #lap_blur = auto_canny(lap_blur)
    circles = cv2.HoughCircles(lap_blur, cv2.HOUGH_GRADIENT, 16, 55, param2=100, minRadius=20, maxRadius=40)
    cimg = None
    #circles = cv2.HoughCircles(lap_blur, cv2.HOUGH_GRADIENT,10, 55, 100, 100, minRadius=20, maxRadius=40)
    if circles is not None:
        cimg = draw_circles(img, circles)
        print("{} circles detected.".format(circles[0].shape[0]))
        # There are some false positives left in the regions containing the numbers.
        # They can be filtered out based on their y-coordinates if your images are aligned to a canonical axis.
        # I'll leave that to you.
    return cimg

def blob (im):
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1
    
    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector_create(params)
    
    # Detect blobs.
    keypoints = detector.detect(im)
    
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Show keypoints
    # Show keypoints
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)
    return im_with_keypoints


def aumentarContraste(img):
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels

    l2 = clahe.apply(l)  # apply CLAHE to the L-channel

    lab = cv2.merge((l2,a,b))  # merge channels
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    return img2

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
 
	# return the edged image
	return edged

def prueba(bgr_img):
    width = 150
    h,w = bgr_img.shape[0],bgr_img.shape[1]
    r = width / float(w)
    dim = (width, int(h * r))
    #bgr_img = aumentarContraste(bgr_img)
    resized = cv2.resize(bgr_img, dim, interpolation = cv2.INTER_AREA)
    cimg = resized
    gray_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.equalizeHist(gray_img)
    img =  cv2.GaussianBlur(gray_img,(9,9),0)
    img = cv2.Canny(img, 50, 100)
    cv2.imshow("",img)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    

def detectC(bgr_img):
    b,g,r = cv2.split(bgr_img)       # get b,g,r
    rgb_img = cv2.merge([r,g,b]) 
    width = 150
    h,w = bgr_img.shape[0],bgr_img.shape[1]
    r = width / float(w)
    dim = (width, int(h * r))
    bgr_img = aumentarContraste(bgr_img)
    bgr_img = cv2.resize(bgr_img, dim, interpolation = cv2.INTER_AREA)
    bgr_img = cv2.fastNlMeansDenoisingColored(bgr_img,None,10,10,7,21)
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    
    
    img =  cv2.GaussianBlur(gray_img,(5,5),11.2)
    gray_img = cv2.equalizeHist(gray_img)
    #img = np.abs(img-255)
    img = cv2.Canny(img, 10, 20)
    
    cv2.imshow("",img)
    k = cv2.waitKey(0)
    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    #circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=35,minRadius=0,maxRadius=0)
    print(circles.shape[1])
    circles = np.uint16(np.around(circles))
    cimg = cv2.cvtColor(gray_img,cv2.COLOR_GRAY2BGR)
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    plt.subplot(121),plt.imshow(rgb_img)
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(cimg)
    plt.title('Hough Transform'), plt.xticks([]), plt.yticks([])
    plt.show()

'''if bgr_img.shape[-1] == 3:           # color image
    b,g,r = cv2.split(bgr_img)       # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb
    bgr_img = aumentarContraste(bgr_img)
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.equalizeHist(gray_img)
else:
    gray_img = bgr_img

#img = cv2.medianBlur(gray_img, 5)
#img = cv2.medianBlur(gray_img, 10)
#img = cv2.bilateralFilter(gray_img,5,50,50)
resized = cv2.resize(gray_img, (100,100), interpolation = cv2.INTER_AREA)
cv2.imshow("",resized)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()'''
#img = cv2.blur(gray_img,(3,3))
#img = kMeans(gray_img,4)
#img = cv2.medianBlur(img, 6)
#img = cv2.blur(img,(3,2))
'''img =  cv2.GaussianBlur(img,(5,5),0)
img = cv2.resize(img, (100,100), interpolation = cv2.INTER_AREA)
cv2.imshow("",img)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
width = 150
h,w = gray_img.shape[0],gray_img.shape[1]
r = width / float(w)
dim = (width, int(h * r))
img = cv2.resize(gray_img, dim, interpolation = cv2.INTER_AREA)

img =  cv2.GaussianBlur(img,(9,9),10)
#img = cv2.Canny(img, 100, 150)

cv2.imshow("",img)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
#circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=35,minRadius=0,maxRadius=0)
print(circles.shape[1])
circles = np.uint16(np.around(circles))
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

plt.subplot(121),plt.imshow(rgb_img)
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(cimg)
plt.title('Hough Transform'), plt.xticks([]), plt.yticks([])
plt.show()'''









'''
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
   cv2.imshow("sad",eye1)
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
            cv2.destroyAllWindows()

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
    if ((part1 > part3) and (part2>part3)): #seguro se puede mejorar
    #if ((part1 > part3) or (part2>part3)) and ((part1>part2)or(part2>part1)):
    #if (part1 +part2) > part3:
        lookSide.append('Right')
    #elif (part2+part3)>part1:
    #elif ((part2 > part1) or (part3>part1)):
    else:
        lookSide.append('Left')
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
'''