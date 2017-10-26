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

def matrixConvolve(m1,m2,i):
    aux = 0
    for (x,y), value in np.ndenumerate(m1):
        aux = aux + (value*m2[x,y])
    if i ==0:
        if aux>255.:
            return 255
        else :
            return int(aux)
    else:
        if int(aux/10)>255:
            return 255
        else :
            return int(aux/10)

def dimensionRC(a,rows):
    x = np.zeros((rows+1),dtype=int)
    init = -int(rows/2)
    limit = int(rows/2) + 1
    if init== 0:
        x[0]=a
    else:
        for i in range(init,limit):
            c = i + int(rows/2)
            x[c]= i + a
    return x

def convolutionFunction(pixelImage,kernel,i):
    dimensionsI = pixelImage.shape
    dimensionK = kernel.shape
    rowsKernel = dimensionK[0] -1
    colsKernel = dimensionK[1] -1
    limitRow = dimensionsI[0] - int(rowsKernel/2) 
    limitCol = dimensionsI[1] - int(colsKernel/2)
    endInitRow = int((rowsKernel/2))
    endInitCol = int(colsKernel/2)
    aux = np.zeros(dimensionsI,dtype=int)
    for (x,y), value in np.ndenumerate(pixelImage):
        c = 0
        if (x < endInitRow or y < endInitCol) or (x >= limitRow or y >= limitCol):
            aux[x,y] = value
        else:
            ixgrid = np.ix_(dimensionRC(x,rowsKernel), dimensionRC(y,colsKernel))
            m1 = pixelImage[ixgrid]
            if i == 0:
                aux[x,y] = matrixConvolve(m1,kernel,0)
            elif i == 1:
                aux[x,y] = matrixConvolve(m1,kernel,1)
    aux = aux.astype(np.uint8)
    return aux

def convolve(inputImage,kernel):
    return convolutionFunction(loadImage(inputImage),kernel,1)
    
def testConvolve(inputImage,kernel):
    compareImages(inputImage,convolve(inputImage,kernel))

#--------------------Gaussian---------------------#
def dimension(sigma):
    a = (2 * (int(np.ceil(3*sigma))))+1
    return a

def arrayGauss1D(sigma):
    return np.zeros((1,dimension(sigma)),dtype=float)
    
def gaussDistribution1D(x,sigma):
    fraction = 1./(np.sqrt(2.*np.pi)*sigma)
    exponential = np.exp(-np.power(x,2.)/2*np.power(sigma,2))
    g = fraction * exponential
    return g

def gaussKernel1D(sigma):
    array1D = arrayGauss1D(sigma)
    f= array1D.shape
    a = -int((f[1]-1)/2)
    c = 0
    d=0
    while (d < f[1]):
        array1D[c,d]=gaussDistribution1D(a,sigma)
        a = a + 1
        d = d + 1
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

def gaussianFilter2D(inputImage,sigma):
    kernel = gaussKernel1D(sigma)
    return convolutionFunction(convolutionFunction(loadImage(inputImage),np.transpose(kernel),0),kernel,0)

def testGaussianFilter2D(inputImage,sigma):
    compareImages(inputImage,gaussianFilter2D(inputImage,sigma))
#-------------------------------------------------#

#-------------------Median Filter-------------------------#

def medianFilter2D(pixelImage,filterSize):
    dimensionsI = pixelImage.shape
    rowsKernel = filterSize[0] -1
    colsKernel = filterSize[1] -1
    limitRow = dimensionsI[0] - int(rowsKernel/2) 
    limitCol = dimensionsI[1] - int(colsKernel/2)
    endInitRow = int((rowsKernel/2))
    endInitCol = int(colsKernel/2)
    aux = np.zeros(dimensionsI,dtype=int)
    for (x,y), value in np.ndenumerate(pixelImage):
        c = 0
        if (x < endInitRow or y < endInitCol) or (x >= limitRow or y >= limitCol):
            aux[x,y] = value
        else:
            ixgrid = np.ix_(dimensionRC(x,rowsKernel), dimensionRC(y,colsKernel))
            m1 = pixelImage[ixgrid]
            aux[x,y] = np.median(m1)
    aux = aux.astype(np.uint8)
    return aux

def testMedianFilter2D(inputImage,filterSize):
    compareImages(inputImage,medianFilter2D(loadImage(inputImage),filterSize))

#-------------------------------------------------#

#----------------------Tests----------------------#

#testWindowLevelContrastEnhancement('lena_gray.bmp',100,20)
#testHistAdapt('lena_gray.bmp',100,200)
#testConvolve('lena_gray.bmp',createKernel(2,3,3))
#testGaussianFilter2D('lena_gray.bmp',1)
#testMedianFilter2D('lena_gray.bmp',(7,7))
#-------------------------------------------------#
