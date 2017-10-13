#Import extensions
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import functools 

#----To show an image
showingImage = functools.partial(plt.imshow,vmin = 0, vmax = 255, cmap= plt.get_cmap('gray'))

#----Function to load the pixels of the image
def loadImage(image):
    return mpimg.imread(image)

#----Function that modify the contrast of pixel
def transferFunction(center,window,a):
    middle = window/2
    minimun = center - middle
    maximun = center + middle
    if a < minimun:
        return 0
    elif a > maximun:
        return 255
    else:
        return 255/window*a-minimun

#----Function that transform the contrast
def histEnhance(inputImage,cenValue,winSize):
    img = loadImage(inputImage)
    data = img.shape
    vfunc = np.vectorize(transferFunction) #apply the tranfer function to each pixel
    img1 = vfunc(cenValue,winSize,img.flatten())
    img2 = np.reshape(img1,data)
    img2 = img2.astype(np.uint8)
    return img2

#----Function that represent outputImage
def showImage(image):
    showingImage(image)
    plt.show()     

#----Function that shows before and after
def compareImages(inputImage,outputImage):
    fig = plt.figure()
    a=fig.add_subplot(1,2,1)
    imgplot = plt.imshow(loadImage(inputImage))
    a.set_title('Before')
    a=fig.add_subplot(1,2,2)
    showingImage(outputImage)
    imgplot.set_clim(0.0,0.7)
    a.set_title('After')
    plt.show()

img = histEnhance('lena.bmp',100,20)
compareImages('lena.bmp',img)


