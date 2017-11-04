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
    ax2.hist(loadImage(inputImage).flatten(),200,range=[0,255])
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
    img1 = vfunc(minValue,maxValue,np.amin(inputImage),inputImage.max(),inputImage.flatten())
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
    limitRow = pixelImage.shape[0] - initRow
    limitCol = pixelImage.shape[1] - initCol
    aux = np.zeros_like(pixelImage)

    for (x,y), value in np.ndenumerate(pixelImage):
        c = 0
        if (x < initRow or y < initCol) or (x >= limitRow or y >= limitCol):
            aux[x,y] = value
        else:
            c = np.sum(kernel*pixelImage[(x-initRow):(x+initRow+1),(y-initCol):(y+initCol+1)])
            if c > 255:
                aux[x,y] = 255
            else:
                aux[x,y] = c
    return aux

def convolve(inputImage,kernel):
    kernel = np.flipud(np.fliplr(kernel))
    a = kernel.sum()
    if a >= 1:
        kernel = kernel/a
    if ((kernel.shape[0]%2) == 0):
       kernel =  np.insert(kernel,kernel.shape[0],0,axis = 0)
    if ((kernel.shape[1]%2) == 0):
        kernel = np.insert(kernel,kernel.shape[1],0,axis = 1)
    return convolutionFunction(inputImage,kernel)
    
def testConvolve(inputImage,kernel):
    compareImages(loadImage(inputImage),convolve(loadImage(inputImage),kernel).astype(np.uint8))

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

def medianFilter2D(pixelImage,filterSize):
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
    aux = aux.astype(np.uint8)
    return aux

def testMedianFilter2D(inputImage,filterSize):
    compareImages(loadImage(inputImage),medianFilter2D(loadImage(inputImage),filterSize))

#-------------------------------------------------#

#------------------HighBoost----------------------#
def aHb(A):
    a = np.zeros((3,3),dtype=int)
    for (x,y),value in np.ndenumerate(a):
        if x == 1 and y == 1:
            a[x,y]= A +8
        else:
            a[x,y]= -1
    return a

def highBoost(inputImage,A,method,parameter):
    a = aHb(A-1)
    ghb1 = convolve(inputImage,a)
    if method == 'gaussian':
        subs = gaussianFilter2D(inputImage,parameter)
    elif method == 'median':
        subs = medianFilter2D(inputImage,parameter)
    ghb2 = inputImage - subs
    ghb = ghb1 + ghb2
    return ghb

#-------------------------------------------------#

#-------------------------------------------------#
def eeType(strElType,strElSize):
    if strElType == 'square':
        if (strElSize[0]%2) == 0 and (strElSize[1]%2) == 0:
            m = np.ones((strElSize[0]+1,strElSize[1]+1),dtype=int)
            for (x,y), value in np.ndenumerate(m):
                if x == 0 or y == strElSize[1]:
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
            m1[0,strElSize[1]] = 0
        else:
            m1 = np.ones(strElSize,dtype=int)
        return m1

def onesInImages(EE):
    r = EE.shape[0]-1
    c = []
    for (x,y),value in np.ndenumerate(EE):
        if value == 1 or value == 255:
            x1 =abs(x - r)
            c.append((x1,y)) 
    return np.array(c,dtype=('int,int'))

def isInside(a,onesImage,onesKernel):
    aux = 255
    for i in onesKernel:
        d = a[0] + i[0],a[1]+i[1]
        v = d in onesImage
        if v == False:
            aux = 0
    return aux

def erodeFunction(image,kernel):
    kernelDimensions = kernel.shape
    onesImage = onesInImages(image).tolist()
    onesKernel = onesInImages(kernel)
    r,c = image.shape[0],image.shape[1]
    rowsKernel = kernelDimensions[0] -1
    colsKernel = kernelDimensions[1] -1
    limitRow = r - int(rowsKernel/2) 
    limitCol = c - int(colsKernel/2)
    initRow = int((rowsKernel/2))
    initCol = int(colsKernel/2)
    aux = np.zeros((image.shape),dtype=int)
    for (x,y),value in np.ndenumerate(image):
        if (x < initRow or y < initCol) or (x >= limitRow or y >= limitCol):
            aux[x,y] = value
        else:
            if value == 255:
                x1 = abs(x-(r-1))
                aux[x,y] = isInside((x1,y),onesImage,onesKernel)
    aux = aux.astype(np.uint8)
    return aux    
    
def erode(inputImage, strElType,strElSize):
    kernel = eeType(strElType,strElSize)
    return erodeFunction(inputImage,kernel)

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
    m = np.zeros((24,24),dtype=int)
    for (x,y), value in np.ndenumerate(m):
        if y == 5 or y == 7 or y == 11 or y ==13 or y == 17:
            m[x,y] = 255
    return m

def testErode(inputImage, strElType,strElSize):
    compareImages(inputImage,erode(inputImage, strElType,strElSize))

def onesInsideImage(image,kernelDimensions):
    r,c = image.shape[0],image.shape[1]
    rowsKernel = kernelDimensions[0] -1
    colsKernel = kernelDimensions[1] -1
    limitRow = r - int(rowsKernel/2) 
    limitCol = c - int(colsKernel/2)
    initRow = int((rowsKernel/2))
    initCol = int(colsKernel/2)
    c = []
    for (x,y),value in np.ndenumerate(image):
        if (x >= initRow and y >= initCol) and (x < limitRow and y < limitCol):
            if value == 255  :
                x1 =abs(x - (r-1))
                c.append((x1,y)) 
    return c

def appendPoint(onesImage,onesEE):
    c = onesImage
    aux = []
    for i in onesImage:
        for j in onesEE:
            d = i[0]+j[0],i[1]+j[1]
            if (d in c) == False:
                #print(i,'+',j,'=',d)
                aux.append(d)
    return np.array(aux,dtype=('int,int'))

def dilateFunction(image,kernel):
    kernelDimensions = kernel.shape
    onesImage = onesInsideImage(image,kernelDimensions)
    onesKernel = onesInImages(kernel)
    r,c = image.shape[0],image.shape[1]
    rowsKernel = kernelDimensions[0] -1
    colsKernel = kernelDimensions[1] -1
    limitRow = r - int(rowsKernel/2) 
    limitCol = c - int(colsKernel/2)
    initRow = int((rowsKernel/2))
    initCol = int(colsKernel/2)
    m = appendPoint(onesImage,onesKernel).tolist()
    aux = np.zeros((image.shape),dtype=int)
    for (x,y),value in np.ndenumerate(image):
        if (x < initRow or y < initCol) or (x >= limitRow or y >= limitCol):
                aux[x,y] = value
        else:
            x1 = abs(x-(r-1))
            d = x1,y
            if (d in m) == True:
                aux[x,y] = 255
            else:
                aux[x,y] = value

    aux = aux.astype(np.uint8)
    return aux    

def dilate(inputImage, strElType,strElSize):
    EE = eeType(strElType,strElSize)
    return dilateFunction(inputImage,EE)

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

#-------------------------------------------------#

#----------Operadores de primera derivada---------#

def gxRoberts():
    gx = np.zeros((2,2),dtype=int)
    gx[0,0] = -1
    gx[1,1] = 1
    gx = np.flipud(np.fliplr(gx))
    aux = np.zeros(((gx.shape[0]+1),(gx.shape[1]+1)),dtype=int)
    aux[1:,1:] = gx
    return aux

def gyRoberts():
    gy = np.zeros((2,2),dtype=int)
    gy[1,0] = 1
    gy[0,1] = -1
    gy = np.flipud(np.fliplr(gy))
    aux = np.zeros(((gy.shape[0]+1),(gy.shape[1]+1)),dtype=int)
    aux[1:,1:] = gy
    return aux

def robertsOperator(inputImage):
    gx = gxRoberts()
    gy = gyRoberts()
    Gx = convolutionFunction(inputImage,gx)
    Gy = convolutionFunction(inputImage,gy)
    return Gx,Gy
    
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
    Gx1 = convolve(inputImage,gx1).astype(np.uint8)
    Gx = convolve(Gx1,gx2).astype(np.uint8)
    Gy = convolve(convolve(inputImage,gy1).astype(np.uint8),gy2).astype(np.uint8)
    return Gx,Gy

def derivatives(inputImage,operator):
    Gx,Gy = 0,0
    if operator == 'Roberts':
        return robertsOperator(inputImage)
    elif operator == 'Prewitt':
        return prewittOperator(inputImage)
    elif operator == 'Sobel':
        return sobelOperator(inputImage)
#-------------------------------------------------#

#----------------------Tests----------------------#

#testWindowLevelContrastEnhancement('lena_gray.bmp',100,20)
#testHistAdapt('lena_gray.bmp',100,200)
#testConvolve('lena_gray.bmp',np.array(([0.1,0.1,0.1],[0.1,0.2,0.1],[0.1,0.1,0.1])))
#testGaussianFilter2D('lena_gray.bmp',1)
#testMedianFilter2D('lena_gray.bmp',(7,7))
#testErode(exampleImage(),'lineh',(1,5))
#testDilate(exampleImage(),'square',(3,3))
#testOpening(exampleImage(),'square',(3,3))
#testClosing(exampleImage(),'square',(3,3))
#-------------------------------------------------#
#gx,gy = derivatives(loadImage('lena_gray.bmp'),'Sobel')
#highBoost(loadImage('lena_gray.bmp'),0,'gaussian',1)