#Import extensions
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from math import e
import functools 

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
    showingImage(loadImage(inputImage))
    a.set_title('Before')
    a=fig.add_subplot(1,2,2)
    showingImage(outputImage)
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

#--Spatial filtering: Smoothing and highlighting--#

def createKernel(maxValue,rows,cols):
    m = np.ones((rows,cols),dtype=int)
    m[int(rows/2),int(cols/2)] = maxValue
    return m

def convolutionFunction(pixelImage,kernel):
    rowsImage,colsImage = pixelImage.shape
    rows,cols = kernel.shape
    rowsKernel = rows -1
    colsKernel = cols -1
    limitRow = rowsImage - int(rowsKernel/2) 
    endLimitRow = rowsImage
    limitCol = colsImage - int(colsKernel/2) 
    endLimitCol = colsImage
    if int(rowsKernel/2)==1 :
        endInitRow = 1
    else:
        endInitRow = int((rowsKernel/2))
    if int(colsKernel/2) == 1:
        endInitCol = 1
    else:
        endInitCol = int(colsKernel/2)
    aux = np.zeros((rowsImage,colsImage),dtype=int)
    for (x,y), value in np.ndenumerate(pixelImage):
        if (x < endInitRow or y < endInitCol) or ((x >= limitRow and x < endLimitRow) or (y >= limitCol and y < endLimitCol)):
            aux[x,y] = value
        else:
            
            q = -1
            q1 = rows
            i =  -1 + x-endInitRow
            i2 = x + endInitRow 
            convolutionValue = 0
            while i < i2 and q < q1:
                i = i +1
                q = q +1
                z = 0 
                z1 = cols
                j = y - endInitCol
                j2 = y + endInitCol +1
                while j < j2  and z < z1 : 
                    count = pixelImage[i,j]*kernel[q,z]   
                    convolutionValue = convolutionValue + count
                    j = j +1
                    z = z+1
                if int(convolutionValue/10) >= 255:
                    aux[x,y] = 255
                else:
                    aux[x,y] = int(convolutionValue/10)
    aux = aux.astype(np.uint8)
    return aux

def convolve(inputImage,kernel):
    return convolutionFunction(loadImage(inputImage),kernel)
    
def testConvolve(inputImage,kernel):
    compareImages(inputImage,convolve(inputImage,kernel))

#--------------------Gaussian---------------------#
def dimension(sigma):
    a = (2 * (int(np.ceil(3*sigma))))+1
    return a

def arrayGauss1D(sigma):
    return np.zeros(dimension(sigma),dtype=float)
    
def gaussDistribution1D(x,sigma):
    fraction = 1./(np.sqrt(2.*np.pi)*sigma)
    exponential = np.exp(-np.power(x,2.)/2*np.power(sigma,2))
    g = fraction * exponential
    return g

def gaussKernel1D(sigma):
    array1D = arrayGauss1D(sigma)
    f= array1D.shape
    b = int(np.ceil(f[0]/2))-1
    a = -b
    c = 0
    while (c < f[0]):
        array1D[c]=gaussDistribution1D(a,sigma)
        a = a + 1
        c = c + 1
    return array1D

def gaussDistributionNxN(x,y,sigma):
    fraction = 1./(2.*np.pi*(np.power(sigma,2.)))
    exponential = np.exp(-((np.power(x,2.) + np.power(y,2.)))/(2*(np.power(sigma,2))))
    g = fraction * exponential
    return g

def gaussKernelNxN(sigma):
    kernel = np.zeros((dimension(sigma),dimension(sigma)),dtype=float)
    f,c = kernel.shape
    a = -int(np.ceil(f/2))
    i = -1
    f = f -1
    c = c -1
    while(i < f):
        a = a +1
        b = -int(np.ceil(c/2))
        i = i +1
        j = 0
        while(j <=c):
            kernel[i,j]= gaussDistributionNxN(a,b,sigma)
            b = b + 1
            j = j +1
    return kernel


#-------------------------------------------------#

#-------------------------------------------------#

#----------------------Tests----------------------#

#testWindowLevelContrastEnhancement('lena_gray.bmp',100,20)
#testHistAdapt('lena_gray.bmp',100,200)
#testConvolve('lena_gray.bmp',createKernel(2,3,3))

#-------------------------------------------------#