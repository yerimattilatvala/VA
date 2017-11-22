# Importamos el paquete de tratamiento de imágenes
import cv2
 
# Definición de rutina para detectar una cara "frontal"
def detect(ruta):
    img = cv2.imread(ruta)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # lee un fichero con un patrón de reconocimiento de caras frontales, PUEDE CAMBIARSE POR OTRO DIFERENTE 
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    
    # Detecta objetos de diferentes tamaños en la imagen, se devuelven como una lista de rectangulos
    scale_factor=1.1
    min_neighbors=3
    min_size=(30,30)
    flags=cv2.CASCADE_SCALE_IMAGE
 
    caras = cascade.detectMultiScale(img, scaleFactor = scale_factor, minNeighbors = min_neighbors,minSize = min_size, flags = flags)
    if len(caras) == 0:
        return [], img
    print(caras)
    #caras[:, 2:] += caras[:, :2]
    return caras, img
 
# Definición de rutina para dibujar recuadros en la imagen y guardarla
def box(caras, img):
    for x1, y1, x2, y2 in caras:
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)
    return img
 
# Rutina principal para procesar todas las imagenes en la ruta con formato 1.jpg, 2.jpg, ...

caras, img = detect('4. Eyes and gaze\Derecha1.jpg')
cv2.imshow('img',box(caras, img))
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()