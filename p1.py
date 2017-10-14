#Import extensions
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import functools 
import matplotlib

#----To show an image
showingImage = functools.partial(plt.imshow,vmin = 0, vmax = 255, cmap= plt.get_cmap('gray'))

#----Function that represent outputImage
def showImage(image):
    showingImage(image)
    plt.show()     

#----Function to load the pixels of the image
def loadImage(image):
    return mpimg.imread(image)

#-------------------------------------------------#

#---------WindowLevelContrastEnhancement----------#

#----Function that modify the contrast of pixel
def transformFunction(center,window,a):
    middle = int(window/2)
    minimun = center - middle
    maximun = center + middle
    if a < minimun:
        return 0
    elif a >= maximun:
        return 255
    else:
        b = int(255/window)
        c = a-minimun
        a = b*c
        return a

#----Function that transform the contrast
def histEnhance(inputImage,cenValue,winSize):
    img = loadImage(inputImage)
    dimsension = img.shape        #----Calcule the array dimensions
    vfunc = np.vectorize(transformFunction)  #----Apply the tranfer function to each pixel
    img1 = vfunc(cenValue,winSize,img.flatten())
    img2 = np.reshape(img1,dimsension)
    img2 = img2.astype(np.uint8)
    return img2

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

#----Function to test the transfrorm
def testWindowLevelContrastEnhancement(image,cenValue,winSize):
    compareImages(image,histEnhance(image,cenValue,winSize))

#-------------------------------------------------#

#-------------------HistAdapt---------------------#
def compareHist(inputImage,outputImage):
    fig = plt.figure()
    ax2 = fig.add_subplot(2, 1, 1)
    ax3 = fig.add_subplot(2, 1, 2)
    ax2.hist(loadImage(inputImage).flatten(),200,range=[0,255])
    ax3.hist(outputImage.flatten(),200,range=[0,255])
    plt.show()

def modifyDinamicRange(gMinNorm,gMaxNorm,gMin,gMax,g):
    a = gMaxNorm - gMinNorm
    b = g - gMin
    c = gMax- gMin
    d = int((a*b)/c)
    return gMinNorm + d
    
def histAdapt(inputImage,minValue,maxValue):
    img = loadImage(inputImage)
    dimsension = img.shape        #----Calcule the array dimensions
    vfunc = np.vectorize(modifyDinamicRange)
    img1 = vfunc(minValue,maxValue,np.amin(img),np.amax(img),img.flatten())
    img2 = np.reshape(img1,dimsension)
    img2 = img2.astype(np.uint8)
    return img2

def testHistAdapt(inputImage,minValue,maxValue):
    compareImages(inputImage,histAdapt(inputImage,minValue,maxValue))
    compareHist(inputImage,histAdapt(inputImage,minValue,maxValue))
#-------------------------------------------------#

#----------------------Tests----------------------#

#testWindowLevelContrastEnhancement('lena.bmp',100,20)
#testHistAdapt('lena.bmp',0,255)

#-------------------------------------------------#



