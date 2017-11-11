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
    plt.imshow(inputImage,cmap= plt.get_cmap('gray'))
    a=fig.add_subplot(1,2,2)
    plt.imshow(outputImage,cmap= plt.get_cmap('gray'))
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
    return img2

def testHistAdapt(inputImage,minValue,maxValue):
    compareImages(loadImage(inputImage),histAdapt(loadImage(inputImage),minValue,maxValue))
    compareHist(loadImage(inputImage),histAdapt(loadImage(inputImage),minValue,maxValue))
#-------------------------------------------------#

def filteredKernel(kernel):
    kernel = np.flipud(np.fliplr(kernel))
    '''a = kernel.sum()
    if a > 1:
        kernel = kernel/a'''
    if ((kernel.shape[0]%2) == 0):
        aux = np.zeros((kernel.shape[0]+1,kernel.shape[1]))
        aux[1:] = kernel
        kernel = aux
    if ((kernel.shape[1]%2) == 0):
        aux = np.zeros((kernel.shape[0],kernel.shape[1]+1))
        aux[0:,1:]=kernel
        kernel = aux
    return kernel
    
#--Spatial filtering: Smoothing and highlighting--#
def convolutionFunction(pixelImage,kernel,operation):
    if operation == 'median':      
        rowsKernel = kernel[0] 
        colsKernel = kernel[1]
        if ((rowsKernel % 2) == 0):
            rowsKernel1 = rowsKernel
            rowsKernel = rowsKernel +1
        else:
            rowsKernel1 = rowsKernel
        if ((colsKernel % 2) == 0):
            colsKernel1  = colsKernel
            colsKernel = colsKernel +1
        else:
            colsKernel1 = colsKernel
    else:                               # Para convolve erode y dilate
        kernel = filteredKernel(kernel)
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
            if operation == 'median':
                c = np.median(image_padded[x:(x+rowsKernel1),y:(y+colsKernel1)])
            elif operation == 'convolve':
                c = np.sum(kernel*image_padded[x:(x+rowsKernel),y:(y+colsKernel)])
            elif operation == 'dilate' :
                window = image_padded[x:(x+rowsKernel),y:(y+colsKernel)]
                c = np.amax((kernel*window))
            elif operation == 'erode':
                    window = image_padded[x:(x+rowsKernel),y:(y+colsKernel)]
                    aux1 = window.copy()
                    aux1[kernel==0] = np.max(window)
                    c=np.min(aux1)
            aux[x,y] = c
    return aux

def convolve(inputImage,kernel):
    return convolutionFunction(inputImage.astype(np.int64),kernel,'convolve')
    
#showImage(abs(convolve(loadImage('lena_gray.bmp'),np.arange(16).reshape((4,4)))))
def testConvolve(inputImage,kernel):
    output = convolve(loadImage(inputImage),kernel)
    compareImages(loadImage(inputImage),output)
    compareImages(loadImage(inputImage),histAdapt(output,0,255)) #hotsAdapt por si valores fuera de rango

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
#showImage(gaussianFilter2D(loadImage('lena_gray.bmp'),2))
#------------------Median Filter------------------#

def medianFilter2D(pixelImage,filterSize): #mejorarlo
    return convolutionFunction(pixelImage.astype(np.int64),filterSize,'median')

def testMedianFilter2D(inputImage,filterSize):
    compareImages(loadImage(inputImage),medianFilter2D(loadImage(inputImage),filterSize))

#-------------------------------------------------#

#------------------HighBoost----------------------#
def highBoost(inputImage,A,method,parameter):
    if method == 'gaussian':
        subs = gaussianFilter2D(inputImage,parameter)
    elif method == 'median':
        subs = medianFilter2D(inputImage,parameter)
    ghb  = np.subtract((A*inputImage.astype(np.int64)),subs)
    return ghb

def testHighBoost(inputImage,A,method,parameter):
    ghb = highBoost(loadImage(inputImage),A,method,parameter)
    #compareImages(loadImage(inputImage),ghb)
    compareImages(loadImage(inputImage),histAdapt(ghb,0,255))
#-------------------------------------------------#
#-------------------------------------------------#
def eeType(strElType,strElSize): 
    if strElType == 'square':
        m = np.ones(strElSize,dtype=int)
        return m
    elif strElType == 'cross' :
        m1 = np.zeros(strElSize,dtype=int)
        for (x,y), value in np.ndenumerate(m1):
            if x == int((strElSize[0]-1)/2) or y == int((strElSize[1]-1)/2):
                m1[x,y] = 1
        return m1
    elif strElType == 'linev' :
        m1 = np.ones(strElSize,dtype=int)
        return m1
    elif strElType == 'lineh' :
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
    a[6:9,6:9] = 1
    a[9,9] = 1
    a[10,9] = 1
    a[10,8] = 1
    a[10,10] = 1
    a[11,9] = 1
    return a

def example3():
    a = np.zeros((16,16),dtype=int)
    a[5,5] = 255
    return a

def dilate(inputImage, strElType, strElSize):
    ee = eeType(strElType,strElSize)
    return convolutionFunction(inputImage.astype(np.int64),ee,'dilate')

def erode(inputImage, strElType, strElSize):
    ee = eeType(strElType,strElSize)
    return convolutionFunction(inputImage,ee,'erode')

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
    compareImages(abs(gx),abs(gy))
    showImage(abs(gx)+abs(gy))
#-------------------------------------------------#

# Canny
def edgeCanny(inputImage,sigma,tlow,thigh):
    smooth = gaussianFilter2D(inputImage,sigma)     # Suavizado gaussiano
    gx,gy = derivatives(smooth,'Sobel')             # sobel
    #Magnitud y orientacion
    magnitude = np.sqrt((gx.astype(np.int64)**2)+(gy.astype(np.int64)**2))
    orientation = np.degrees(np.arctan2(gy,gx))
    #Supresión no máximos
    image_padedd = np.zeros((inputImage.shape[0]+2,inputImage.shape[1]+2))
    image_padedd[1:-1,1:-1] = magnitude
    In = np.zeros_like(magnitude)
    # Ajustar ángulos a 0º,45º,90º,135º
    for (x,y), value in np.ndenumerate(orientation):
        if orientation[x,y]  < 0:
            orientation[x,y] =orientation[x,y]+180
        if orientation[x,y] > 112.5 and orientation[x,y] < 157.5:
            orientation[x,y] = 135
        elif orientation[x,y] > 67.5 and orientation[x,y] < 112.5:
            orientation[x,y] = 90
        elif orientation[x,y] > 22.5 and orientation[x,y] < 67.5:
            orientation[x,y] = 45
        elif orientation[x,y] > 157.5 or orientation[x,y] < 22.5:
            orientation[x,y] = 0
        
    for (x,y), value in np.ndenumerate(orientation):
        window = image_padedd[x:x+3,y:y+3]
        if (value ==135.0):
            n1 = window[0,0]
            n2 = window[2,2]
        elif (value == 90.0)  :
            n1 = window[0,1]
            n2 = window[2,1]
        elif (value == 45.0):
            n1 = window[0,2]
            n2 = window[2,0]
        elif (value == 0.0):
            n1 = window[1,0]
            n2 = window[1,2]
        if (magnitude[x,y] > n1 and magnitude[x,y] > n2):
            In[x,y] = magnitude[x,y]
    #Fin supresión no maxima
    #Proceso de histeresis
    visitados = []
    H = np.zeros_like(In)
    for (x,y),value in np.ndenumerate(orientation):
        if (In[x,y] > thigh) and (((x,y) in visitados) == False):
            H[x,y] = In[x,y]
            if value==0.0 :
                k1 = x-1
                k2 = x+1
                l1 = y
                l2 = y
                while((k1 >0) and (k2 >0) and (l1 >0) and (l2 >0) and (k1 < In.shape[0]) and (k2 < In.shape[0]) and (l1 < In.shape[1]) and (l2 < In.shape[1]) and (In[k1,l1]>tlow) and (In[k2,l2]>tlow)):
                    H[k1,l1] = In[x,y]
                    H[k2,l2] = In[x,y]
                    visitados.append((k1,l1))
                    visitados.append((k2,l2))
                    k1 = k1-1
                    k2 = k2+1
            elif value == 45.0 :
                k1 = x-1
                k2 = x+1
                l1 = y-1
                l2 = y+1
                while((k1 >0) and (k2 >0) and (l1 >0) and (l2 >0) and (k1 < In.shape[0]) and (k2 < In.shape[0]) and (l1 < In.shape[1]) and (l2 < In.shape[1]) and (In[k1,l1]>tlow) and (In[k2,l2]>tlow)):
                    H[k1,l1] = In[x,y]
                    H[k2,l2] = In[x,y]
                    visitados.append((k1,l1))
                    visitados.append((k2,l2))
                    k1 = k1-1
                    k2 = k2+1
                    l1 = l1-1
                    l2=l2+1
            elif value == 90.0 :
                k1 = x
                k2 = x
                l1 = y-1
                l2 = y+1
                while((k1 >0) and (k2 >0) and (l1 >0) and (l2 >0) and (k1 < In.shape[0]) and (k2 < In.shape[0]) and (l1 < In.shape[1]) and (l2 < In.shape[1]) and (In[k1,l1]>tlow) and (In[k2,l2]>tlow)):
                    H[k1,l1] = In[x,y]
                    H[k2,l2] = In[x,y]
                    visitados.append((k1,l1))
                    visitados.append((k2,l2))
                    l1 = l1-1
                    l2 = l2+1
            elif value == 135.0:
                k1 = x-1
                k2 = x+1
                l1 = y+1
                l2 =y-1
                while((k1 >0) and (k2 >0) and (l1 >0) and (l2 >0) and (k1 < In.shape[0]) and (k2 < In.shape[0]) and (l1 < In.shape[1]) and (l2 < In.shape[1]) and (In[k1,l1]>tlow) and (In[k2,l2]>tlow)):
                    H[k1,l1] = In[x,y]
                    H[k2,l2] = In[x,y]
                    visitados.append((k1,l1))
                    visitados.append((k2,l2))
                    k1 = k1-1
                    k2 = k2+1 
                    l1 = l1+1
                    l2 = l2-1
    return H 
        
#edgeCanny(loadImage('lena_gray.bmp'),0.625,30,50)

# Harris
#----------------------Tests----------------------#
#testWindowLevelContrastEnhancement('lena_gray.bmp',100,20)
#testHistAdapt('lena_gray.bmp',100,200)
#testConvolve('lena_gray.bmp',np.array(([1,1,1],[1,11,1],[1,1,1])).reshape((3,3)))
#testGaussianFilter2D('lena_gray.bmp',3)
#testMedianFilter2D('lena_gray.bmp',(5,4))
#testHighBoost('lena_gray.bmp',3,'median',(11,11))
#testErode(exampleImage2(),'lineh',(1,2))
#testDilate(exampleImage(),'square',(3,3))
#testOpening(exampleImage(),'square',(3,3))
#testClosing(exampleImage(),'square',(3,3))
#testDerivatives('lena_gray.bmp','Roberts')
#-------------------------------------------------#