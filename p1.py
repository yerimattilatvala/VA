#Import extensions
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from math import e
import functools 

#----To show an image
showingImage = functools.partial(plt.imshow, cmap= plt.get_cmap('gray'))

#----Function that represent outputImage
def showImage(image):
    showingImage(image)
    plt.show()     

#----Function to load the pixels of the image
def loadImage(image):
    data = mpimg.imread(image)
    return data

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
    dimsension = inputImage.shape        #----Calcule the array dimensions
    vfunc = np.vectorize(transformFunction)  #----Apply the tranfer function to each pixel
    img1 = vfunc(cenValue,winSize,inputImage.flatten())
    img2 = np.reshape(img1,dimsension)
    img2 = img2.astype(np.uint8)
    return img2

#----Function that shows before and after
def compareImages(inputImage,outputImage):
    fig = plt.figure()
    a=fig.add_subplot(1,2,1)
    showingImage(inputImage)
    a=fig.add_subplot(1,2,2)
    showingImage(outputImage)
    plt.show()

#----Function to test the transfrorm
def testWindowLevelContrastEnhancement(image,cenValue,winSize):
    compareImages(loadImage(image),histEnhance(oadImage(image),cenValue,winSize))

#-------------------------------------------------#

#-------------------HistAdapt---------------------#
def compareHist(inputImage,outputImage):
    fig = plt.figure()
    ax2 = fig.add_subplot(2, 1, 1)
    ax3 = fig.add_subplot(2, 1, 2)
    ax2.hist(inputImage.flatten(),200,range=[0,255])
    ax3.hist(outputImage.flatten(),200,range=[0,255])
    plt.show()

def modifyDinamicRange(gMinNorm,gMaxNorm,gMin,gMax,g):
    a = (gMaxNorm - gMinNorm)
    b = (g - gMin)
    c = (gMax- gMin)
    d = int((a*b)/c)
    return gMinNorm + d
    
def histAdapt(inputImage,minValue,maxValue):
    dimsension = inputImage.shape        #----Calcule the array dimensions
    vfunc = np.vectorize(modifyDinamicRange)
    img1 = vfunc(minValue,maxValue,np.min(inputImage),np.max(inputImage),inputImage.flatten())
    img2 = np.reshape(img1,dimsension)
    img2 = img2.astype(np.uint8)
    return img2

def testHistAdapt(inputImage,minValue,maxValue):
    compareImages(loadImage(inputImage),histAdapt(loadImage(inputImage),minValue,maxValue))
    compareHist(inputImage,histAdapt(loadImage(inputImage),minValue,maxValue))
#-------------------------------------------------#

#--Spatial filtering: Smoothing and highlighting--#
def convolutionFunction(pixelImage,kernel):
    rowsKernel = kernel.shape[0] 
    colsKernel = kernel.shape[1] 
    initRow = int((rowsKernel-1)/2)
    initCol = int((colsKernel-1)/2)
    image_padded = np.zeros((pixelImage.shape[0] + (rowsKernel -1 ), pixelImage.shape[1] + (colsKernel-1)))
    if (colsKernel -1) == 0 and (rowsKernel -1 ) == 0: #caso kernel 1x1
          image_padded = pixelImage
    elif (colsKernel -1) == 0: 
        image_padded[initRow:-initRow] = pixelImage
    elif (rowsKernel -1 ) == 0:
        image_padded[0:,initCol:-initCol] = pixelImage
    else:
         image_padded[initRow:-initRow,initCol:-initCol] = pixelImage
    aux = np.zeros_like(pixelImage)

    for x in range(pixelImage.shape[0]):
        for y in range(pixelImage.shape[1]):
            c = np.sum(kernel*image_padded[x:(x+rowsKernel),y:(y+colsKernel)])
            if c > 255:
                c = 255
            elif c < 0:
                c = 0
            aux[x,y] = c
    return aux

def convolve(inputImage,kernel):
    kernel = np.flipud(np.fliplr(kernel))
    a = kernel.sum()
    if a > 1:
        kernel = kernel/a
    if ((kernel.shape[0]%2) == 0):
        kernel = np.insert(kernel,kernel.shape[0],0,axis=0)
    if ((kernel.shape[1]%2) == 0):
        kernel = np.insert(kernel,kernel.shape[1],0,axis=1)
    return convolutionFunction(inputImage,kernel)
    
def testConvolve(inputImage,kernel):
    compareImages(loadImage(inputImage),convolve(loadImage(inputImage),kernel))

#--------------------Gaussian---------------------#
def dimension(sigma):
    a = (2 * (int(np.ceil(3*sigma))))+1
    return a

def arrayGauss1D(sigma):
    return np.zeros((1,dimension(sigma)),dtype=float)
    
def gaussDistribution1D(x,sigma):
    fraction = 1./(np.sqrt(2.*np.pi)*sigma)
    exponential = np.exp(-((x**2)/(2.0*sigma**2)))
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
    A = convolve(inputImage,np.transpose(kernel))
    return convolve(A,kernel)

def testGaussianFilter2D(inputImage,sigma):
    compareImages(loadImage(inputImage),gaussianFilter2D(loadImage(inputImage),sigma))
#-------------------------------------------------#

#------------------Median Filter------------------#

def medianFilter2D(pixelImage,filterSize): #mejorarlo
    rowsKernel = filterSize[0] 
    colsKernel = filterSize[1] 
    initRow = int((rowsKernel-1)/2)
    initCol = int((colsKernel-1)/2)
    limitRow = pixelImage.shape[0] - initRow
    limitCol = pixelImage.shape[1] - initCol
    aux = np.zeros_like(pixelImage)

    for (x,y), value in np.ndenumerate(pixelImage):
        if (x < initRow or y < initCol) or (x >= limitRow or y >= limitCol):
            aux[x,y] = value
        else:
            aux[x,y] = np.median(pixelImage[(x-initRow):(x+initRow+1),(y-initCol):(y+initCol+1)])
    return aux

def testMedianFilter2D(inputImage,filterSize):
    compareImages(loadImage(inputImage),medianFilter2D(loadImage(inputImage),filterSize))

#-------------------------------------------------#

#------------------HighBoost----------------------#
def highBoost(inputImage,A,method,parameter):
    if method == 'gaussian':
        subs = gaussianFilter2D(inputImage,parameter)
    elif method == 'median':
        subs = medianFilter2D(inputImage,parameter)
    ghb  = np.subtract((A*inputImage.astype(np.int64)).astype(np.int64),subs)
    return ghb

#-------------------------------------------------#

#-------------------------------------------------#
def eeType(strElType,strElSize): 
    if strElType == 'square':
        if (strElSize[0]%2) == 0 and (strElSize[1]%2) == 0:
            m = np.ones((strElSize[0]+1,strElSize[1]+1),dtype=int)
            for (x,y), value in np.ndenumerate(m):
                if x == strElSize[0] or y == strElSize[1]:
                    m[x,y] = 0
        else: 
            m = np.ones(strElSize,dtype=int)
        return m
    elif strElType == 'cross' :
        m1 = np.zeros(strElSize,dtype=int)
        for (x,y), value in np.ndenumerate(m1):
            if x == int((strElSize[0]-1)/2) or y == int((strElSize[1]-1)/2):
                m1[x,y] = 1
        return m1
    elif strElType == 'linev' :
        if (strElSize[0]%2) == 0:
            m1 = np.ones((strElSize[0]+1,strElSize[1]),dtype=int)
            m1[0]= 0
        else:
            m1 = np.ones(strElSize,dtype=int)
        return m1
    elif strElType == 'lineh' :
        print()
        if (strElSize[1]%2) == 0:
            m1 = np.ones((strElSize[0],strElSize[1]+1),dtype=int)
            m1[0,0] = 0
        else:
            m1 = np.ones(strElSize,dtype=int)
        return m1

def exampleImage():
    a = np.zeros((16,16),dtype=int)
    a[2,3]=255
    a[2,4]=255
    a[2,5]=255
    a[3,2] = 255
    a[3,3]=255
    a[3,4]=255
    a[3,5]=255
    a[3,6] = 255
    a[4,2] = 255
    a[4,3]=255
    a[4,4]=255
    a[4,5]=255
    a[4,6] = 255
    a[4,11]=255
    a[4,12]=255
    a[4,13] = 255
    a[5,2] = 255
    a[5,3]=255
    a[5,4]=255
    a[5,5]=255
    a[5,10] = 255
    a[5,11]=255
    a[5,12]=255
    a[5,13] = 255
    a[6,3]=255
    a[6,4]=255
    a[6,9] = 255
    a[6,10] = 255
    a[6,11]=255
    a[6,12]=255
    a[6,13] = 255
    a[7,9] = 255
    a[7,10] = 255
    a[7,11]=255
    a[7,12]=255
    a[7,8] = 255
    a[8,9] = 255
    a[8,10] = 255
    a[8,11]=255
    a[8,7]=255
    a[8,8] = 255
    a[9,9] = 255
    a[9,10] = 255
    a[9,6]=255
    a[9,7]=255
    a[9,8] = 255
    a[10,9] = 255
    a[10,5] = 255
    a[10,6]=255
    a[10,7]=255
    a[10,8] = 255
    a[11,4] = 255
    a[11,5] = 255
    a[11,6]=255
    a[11,7]=255
    a[11,8] = 255
    a[12,7]=255
    a[12,8] = 255
    a[12,4] = 255
    a[12,5] = 255
    a[12,6]=255
    a[12,10]=255
    a[12,11] = 255
    a[12,9] =255
    a[13,7]=255
    a[13,8] = 255
    a[13,4] = 255
    a[13,5] = 255
    a[13,6]=255
    a[13,10]=255
    a[13,11] = 255
    a[13,9] =255
    a[14,7]=255
    a[14,8] = 255
    a[14,5] = 255
    a[14,6]=255
    a[14,10]=255
    a[14,9] =255
    return a

def exampleImage2():
    a = np.zeros((16,16),dtype=int)
    a[1:3,2:4] = 255
    a[5:7,5:7] = 255
    return a

def dilate(inputImage,strElType,strElSize):
    ee = eeType(strElType,strElSize)
    rowsKernel = ee.shape[0] 
    colsKernel = ee.shape[1] 
    initRow = int((rowsKernel-1)/2)
    initCol = int((colsKernel-1)/2)
    image_padded = np.zeros((inputImage.shape[0] + (rowsKernel -1 ), inputImage.shape[1] + (colsKernel-1)))

    output = np.zeros_like(inputImage)

    for x in range(inputImage.shape[0]):
        for y in range(inputImage.shape[1]):
            if inputImage[x,y] == 255 or inputImage[x,y] ==1:
                image_padded[x:(x+rowsKernel),y:(y+colsKernel)] = ee*255 + image_padded[x:(x+rowsKernel),y:(y+colsKernel)]

    if (colsKernel -1) == 0 and (rowsKernel -1 ) == 0: #caso kernel 1x1
          output = image_padded 
    elif (colsKernel -1) == 0: 
        output = image_padded[initRow:-initRow]
    elif (rowsKernel -1 ) == 0:
        output = image_padded[0:,initCol:-initCol]
    else:
         output = image_padded[initRow:-initRow,initCol:-initCol]
    
    return output.astype(np.uint8)

def erode(inputImage,strElType,strElSize):
    ee = eeType(strElType,strElSize)
    rowsKernel = ee.shape[0] 
    colsKernel = ee.shape[1] 
    initRow = int((rowsKernel-1)/2)
    initCol = int((colsKernel-1)/2)
    image_padded = np.zeros((inputImage.shape[0] + (rowsKernel -1 ), inputImage.shape[1] + (colsKernel-1)))
    if (colsKernel -1) == 0 and (rowsKernel -1 ) == 0: #caso kernel 1x1
          image_padded = inputImage
    elif (colsKernel -1) == 0: 
        image_padded[initRow:-initRow] = inputImage
    elif (rowsKernel -1 ) == 0:
        image_padded[0:,initCol:-initCol] = inputImage
    else:
         image_padded[initRow:-initRow,initCol:-initCol] = inputImage
    
    output = np.zeros_like(inputImage)
    for x in range(inputImage.shape[0]):
        for y in range(inputImage.shape[1]):
            if inputImage[x,y] == 255 or inputImage[x,y] ==1:
                #image_padded[x:(x+rowsKernel),y:(y+colsKernel)] = ee*255 + image_padded[x:(x+rowsKernel),y:(y+colsKernel)]
                if (np.allclose(ee*255,image_padded[x:(x+rowsKernel),y:(y+colsKernel)])) ==True:
                    output[x,y] = 255
    return output.astype(np.uint8)

def testErode(inputImage, strElType,strElSize):
    compareImages(inputImage,erode(inputImage, strElType,strElSize))

def testDilate(inputImage,strElType,strElSize):
    compareImages(inputImage,dilate(inputImage,strElType,strElSize))

def opening(inputImage, strElType,strElSize):
    return dilate(erode(inputImage,strElType,strElSize,),strElType,strElSize)

def testOpening(inputImage,strElType,strElSize):
    compareImages(inputImage,opening(inputImage,strElType,strElSize))

def closing(inputImage, strElType,strElSize):
    return erode(dilate(inputImage,strElType,strElSize,),strElType,strElSize)

def testClosing(inputImage,strElType,strElSize):
    compareImages(inputImage,closing(inputImage,strElType,strElSize))

def tophatFilter(inputImage,strElType,strElSize,mode):
    if mode == 'white':
        img = np.subtract(inputImage, opening(inputImage,strElType,strElSize).astype(np.int64))
    elif mode == 'black':
        img = np.subtract(inputImage,closing(inputImage,strElType,strElSize).astype(np.int64))
    return img
#----------Operadores de primera derivada---------#

def gxRoberts():
    gx = np.zeros((2,2),dtype=int)
    gx[0,0] = -1
    gx[1,1] = 1
    return gx

def gyRoberts():
    gy = np.zeros((2,2),dtype=int)
    gy[1,0] = 1
    gy[0,1] = -1
    return gy

def robertsOperator(inputImage):
    gx = gxRoberts()
    gy = gyRoberts()
    Gx = convolve(inputImage,gx)
    Gy = convolve(inputImage,gy)
    return Gy,Gx
    
def gxPrewitt():
    gx1 = np.ones((3,1),dtype=int)
    gx2 = np.ones((1,3),dtype=int)
    gx2[0,0] = -1
    gx2[0,1] = 0 
    return gx1,gx2

def gyPrewitt():
    gy2 = np.ones((1,3),dtype=int)
    gy1 = np.ones((3,1),dtype=int)
    gy1[0,0] = -1
    gy1[1,0] = 0

    return gy1,gy2

def prewittOperator(inputImage):
    gx1,gx2 = gxPrewitt()
    gy1,gy2 = gyPrewitt()
    Gx = convolve(convolve(inputImage,gx1),gx2)
    Gy = convolve(convolve(inputImage,gy1),gy2)
    return Gx,Gy

def gxSobel():
    gx1 = np.ones((3,1),dtype=int)
    gx1[1,0]=2
    gx2 = np.zeros((1,3),dtype = int)
    gx2[0,0] = -1
    gx2[0,2] = 1
    return gx1,gx2

def gySobel():
    gx1 = np.ones((1,3),dtype=int)
    gx1[0,1]=2
    gx2 = np.zeros((3,1),dtype = int)
    gx2[0,0] = 1
    gx2[2,0] = -1
    return gx2,gx1

def sobelOperator(inputImage):
    gx1,gx2 = gxSobel()
    gy1,gy2 = gySobel()
    Gx1 = convolve(inputImage,gx1)
    Gx = convolve(Gx1,gx2)
    Gy = convolve(convolve(inputImage,gy1),gy2)
    return Gx,Gy

def centralDiffOperator(inputImage): #revisar
    k = np.array(([-1,0,1])).reshape(1,3)
    Gx = convolve(inputImage,k)
    Gy = convolve(inputImage,np.transpose(k))
    return Gx,Gy

def derivatives(inputImage,operator):
    if operator == 'Roberts':
        return robertsOperator(inputImage)
    elif operator == 'Prewitt':
        return prewittOperator(inputImage)
    elif operator == 'Sobel':
        return sobelOperator(inputImage)
    elif operator == 'CentralDiff':
        return centralDiffOperator(inputImage)

def testDerivatives(inputImage,operator):
    gx,gy = derivatives(loadImage(inputImage),operator)
    compareImages(gx,gy)
    showImage(gx+gy)
#-------------------------------------------------#

#----------------------Tests----------------------#
#testWindowLevelContrastEnhancement('lena_gray.bmp',100,20)
#testHistAdapt('lena_gray.bmp',100,200)
#testConvolve('lena_gray.bmp',np.array(([0.1,0.1,0.1],[0.1,0.2,0.1],[0.1,0.1,0.1])))
#testGaussianFilter2D('lena_gray.bmp',5)
#testMedianFilter2D('lena_gray.bmp',(7,7))
#testErode(exampleImage(),'lineh',(1,5))
#testDilate(exampleImage(),'square',(3,3))
#testOpening(exampleImage(),'square',(3,3))
#testClosing(exampleImage(),'square',(3,3))
#testDerivatives('lena_gray.bmp','Roberts')
#-------------------------------------------------#

#ghb = highBoost(loadImage('lena_gray.bmp'),1,'median',(5,5))
#compareImages(loadImage('lena_gray.bmp'),histAdapt(ghb,0,255))