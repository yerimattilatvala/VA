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
    showingImage(inputImage)
    a=fig.add_subplot(1,2,2)
    showingImage(outputImage)
    plt.show()

#----Function to test the transfrorm
def testWindowLevelContrastEnhancement(image,cenValue,winSize):
    compareImages(loadImage(image),histEnhance(image,cenValue,winSize))

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
    compareImages(loadImage(inputImage),histAdapt(inputImage,minValue,maxValue))
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

def convolutionFunction(pixelImage,kernel):
    if (type(kernel[0,0])) == np.int32:
        i = 1
    elif (type(kernel[0,0])) == np.float64:
        i = 0
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
            aux[x,y] = matrixConvolve(m1,kernel,i)
    aux = aux.astype(np.uint8)
    return aux

def convolve(inputImage,kernel):
    print(kernel)
    kernel = np.rot90(np.rot90(kernel))
    print(kernel)
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
    return convolve(convolve(inputImage,np.transpose(kernel)),kernel)

def testGaussianFilter2D(inputImage,sigma):
    compareImages(loadImage(inputImage),gaussianFilter2D(loadImage(inputImage),sigma))
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
    compareImages(loadImage(inputImage),medianFilter2D(loadImage(inputImage),filterSize))

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

def gxPrewitt():
    gx1 = np.ones((3,1),dtype=int)
    gx2 = np.array((-1,0,1),dtype=int)
    return gx1,gx2

def gyPrewitt():
    gy2 = np.ones((1,3),dtype=int)
    gy1 = np.array(([-1],[0],[1]),dtype=int)
    return gy1,gy2

def prewittOperator():
    gx1,gx2 = gxPrewitt()
    gy1.gy2 = gyPrewitt()
    Gx = convolve(convolve(inputImage,gx1),gx2)
    Gy = convolve(convolve(inputImage,gy1),gy2)
    compareImages(Gx,Gy)

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
    Gx = convolve(convolve(inputImage,gx1),gx2)
    Gy = convolve(convolve(inputImage,gy1),gy2)
    compareImages(Gx,Gy)

#-------------------------------------------------#

#----------------------Tests----------------------#

#testWindowLevelContrastEnhancement('lena_gray.bmp',100,20)
#testHistAdapt('lena_gray.bmp',100,200)
#testConvolve('lena_gray.bmp',createKernel(2,3,3))
#testGaussianFilter2D('lena_gray.bmp',2)
#testMedianFilter2D('lena_gray.bmp',(7,7))
#testErode(exampleImage(),'lineh',(1,5))
#testDilate(exampleImage(),'square',(3,3))
#testOpening(exampleImage(),'square',(3,3))
#testClosing(exampleImage(),'square',(3,3))
#-------------------------------------------------#

sobelOperator(loadImage('lena_gray.bmp'))