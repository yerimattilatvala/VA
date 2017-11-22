#Import extensions
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from math import e
import functools 


showingImage = functools.partial(plt.imshow, vmin = 0, vmax = 255,cmap= plt.get_cmap('gray'))

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

#----Function to test the transfrorm
def testWindowLevelContrastEnhancement(image,cenValue,winSize):
    fig = plt.figure()
    a=fig.add_subplot(221)
    a.imshow(image,cmap= plt.get_cmap('gray'))
    a=fig.add_subplot(222)
    a.hist(image.flatten(),200,range=[0,255])
    a = fig.add_subplot(223)
    output = histEnhance(image,cenValue,winSize)
    a.imshow(output,cmap= plt.get_cmap('gray'))
    a=fig.add_subplot(224)
    a.hist(output.flatten(),200,range=[0,255])
    plt.show()

#-------------------------------------------------#

#-------------------HistAdapt---------------------#
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
    fig = plt.figure()
    a=fig.add_subplot(221)
    showingImage(inputImage)
    a=fig.add_subplot(222)
    a.hist(inputImage.flatten(),200,range=[0,255])
    a = fig.add_subplot(223)
    output = histAdapt(inputImage,minValue,maxValue)
    showingImage(output)
    a=fig.add_subplot(224)
    a.hist(output.flatten(),200,range=[0,255])
    plt.show()

#-------------------------------------------------#

def filteredKernel(kernel):
    kernel = np.flipud(np.fliplr(kernel))
    '''if operation == 'convolve':
        a = kernel.sum()
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
    return convolutionFunction(inputImage.astype(np.float64),kernel,'convolve')
    
def testConvolve(inputImage,kernel):
    output = convolve(loadImage(inputImage),kernel)
    fig = plt.figure()
    a=fig.add_subplot(121)
    a.set_title('Original')
    a.imshow(loadImage(inputImage),cmap= plt.get_cmap('gray'))
    a = fig.add_subplot(122)
    a.set_title('Filtered')
    a.imshow(histAdapt(output,0,255), vmin = 0, vmax=255,cmap= plt.get_cmap('gray'))
    plt.show()

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
    A = convolve(inputImage,kernel)
    return convolve(A,np.transpose(kernel))

def testGaussianFilter2D(image,sigma):
    #image = loadImage(inputImage)
    output = gaussianFilter2D(image,sigma)
    print(gaussKernel1D(sigma))
    print(np.min(output),np.max(output))
    fig = plt.figure()
    a=fig.add_subplot(121)
    a.set_title('Original')
    a.imshow(image,cmap= plt.get_cmap('gray'))
    a = fig.add_subplot(122)
    a.set_title('Gaussian Filter')
    a.imshow(output,cmap= plt.get_cmap('gray'))
    plt.show()
#-------------------------------------------------#

#------------------Median Filter------------------#

def medianFilter2D(pixelImage,filterSize): #mejorarlo
    return convolutionFunction(pixelImage.astype(np.int64),filterSize,'median')

def testMedianFilter2D(inputImage,filterSize):
    image = loadImage(inputImage)
    output = medianFilter2D(image,filterSize)
    fig = plt.figure()
    a=fig.add_subplot(121)
    a.set_title('Original')
    a.imshow(image,cmap= plt.get_cmap('gray'))
    a = fig.add_subplot(122)
    a.set_title('Median Filter')
    a.imshow(output,cmap= plt.get_cmap('gray'))
    plt.show()
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
    image = loadImage(inputImage)
    output = highBoost(image,A,method,parameter)
    print(np.min(output),np.max(output))
    fig = plt.figure()
    a=fig.add_subplot(121)
    a.set_title('Original')
    a.imshow(image,cmap= plt.get_cmap('gray'))
    a = fig.add_subplot(122)
    if method == 'gaussian':
        a.set_title('HighBoost (method gaussian)')
    else:
        a.set_title('HighBoost (method median)')
    a.imshow(histAdapt(output,0,255),vmin = 0, vmax = 255,cmap= plt.get_cmap('gray'))
    plt.show()
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
    return convolutionFunction(inputImage,ee,'dilate')

def erode(inputImage, strElType, strElSize):
    ee = eeType(strElType,strElSize)
    return convolutionFunction(inputImage,ee,'erode')

def testErode(inputImage, strElType,strElSize):
    output = erode(inputImage,strElType,strElSize)
    fig = plt.figure()
    a=fig.add_subplot(121)
    a.set_title('Original')
    a.imshow(inputImage,cmap= plt.get_cmap('gray'))
    a = fig.add_subplot(122)
    a.set_title('Erode')
    a.imshow(output,cmap= plt.get_cmap('gray'))
    plt.show()

def testDilate(inputImage,strElType,strElSize):
    output = dilate(inputImage,strElType,strElSize)
    fig = plt.figure()
    a=fig.add_subplot(121)
    a.set_title('Original')
    a.imshow(inputImage,cmap= plt.get_cmap('gray'))
    a = fig.add_subplot(122)
    a.set_title('Dilate')
    a.imshow(output,cmap= plt.get_cmap('gray'))
    plt.show()

def opening(inputImage, strElType,strElSize):
    return dilate(erode(inputImage,strElType,strElSize,),strElType,strElSize)

def testOpening(inputImage,strElType,strElSize):
    output = opening(inputImage,strElType,strElSize)
    fig = plt.figure()
    a=fig.add_subplot(121)
    a.set_title('Original')
    a.imshow(inputImage,cmap= plt.get_cmap('gray'))
    a = fig.add_subplot(122)
    a.set_title('Opening')
    a.imshow(output,cmap= plt.get_cmap('gray'))
    plt.show()

def closing(inputImage, strElType,strElSize):
    return erode(dilate(inputImage,strElType,strElSize,),strElType,strElSize)

def testClosing(inputImage,strElType,strElSize):
    output = closing(inputImage,strElType,strElSize)
    fig = plt.figure()
    a=fig.add_subplot(121)
    a.set_title('Original')
    a.imshow(inputImage,cmap= plt.get_cmap('gray'))
    a = fig.add_subplot(122)
    a.set_title('Closing')
    a.imshow(output,cmap= plt.get_cmap('gray'))
    plt.show()

def tophatFilter(inputImage,strElType,strElSize,mode):
    if mode == 'white':
        img = np.subtract(inputImage, opening(inputImage,strElType,strElSize))
    elif mode == 'black':
        img = np.subtract(closing(inputImage,strElType,strElSize),inputImage)
    return img

def testTopHat(inputImage,strElType,strElSize,mode):
    output = tophatFilter(inputImage,strElType,strElSize,mode)
    fig = plt.figure()
    a=fig.add_subplot(121)
    a.set_title('Original')
    a.imshow(inputImage,cmap= plt.get_cmap('gray'))
    a = fig.add_subplot(122)
    if mode == 'white':
        a.set_title('TopHat (mode white)')
    else:
        a.set_title('TopHat (mode black)')
    a.imshow(output,cmap= plt.get_cmap('gray'))
    plt.show()
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

def testDerivatives(image,operator):
    #image = loadImage(inputImage)
    gx,gy = derivatives(image,operator)
    fig = plt.figure()
    a=fig.add_subplot(221)
    a.set_title('Original')
    a.imshow(image,cmap= plt.get_cmap('gray'))
    a = fig.add_subplot(222)
    a.set_title('gx')
    a.imshow(abs(gx),cmap= plt.get_cmap('gray'))
    a = fig.add_subplot(223)
    a.set_title('gy')
    a.imshow(abs(gy),cmap= plt.get_cmap('gray'))
    a = fig.add_subplot(224)
    a.set_title('gx+gy')
    a.imshow((abs(gx)+abs(gy)),cmap= plt.get_cmap('gray'))
    plt.show()
#-------------------------------------------------#

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
        if value ==135.0 and (n1 > magnitude[x,y] or n2 > magnitude[x,y]) :
            In[x,y] = 0
        else:
            In[x,y] = magnitude[x,y]
        '''if (magnitude[x,y] > n1 and magnitude[x,y] > n2):
            In[x,y] = magnitude[x,y]'''
    #Fin supresión no maxima
    #Proceso de histeresis
    '''visitados = []
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
    '''
    return In
        
def testEdgeCanny(inputImage,sigma,tlow,thigh):
    image = loadImage(inputImage)
    output=edgeCanny(image,sigma,tlow,thigh)
    fig = plt.figure()
    a=fig.add_subplot(121)
    a.set_title('Original')
    a.imshow(image,cmap= plt.get_cmap('gray'))
    a = fig.add_subplot(122)
    a.set_title('Canny')
    a.imshow(output,cmap= plt.get_cmap('gray'))
    plt.show()
#-------------------------------------------------#

#-------------------------------------------------#
# Harris
def cornerHarris(inputImage,sigmaD,sigmaI,t):
    gD = gaussianFilter2D(inputImage,sigmaD)
    gx,gy = derivatives(gD,'Sobel')
    Ix2 = gx**2
    Iy2 = gy**2
    IxIy = gx*gy
    gIx2 = gaussianFilter2D(Ix2,sigmaI)
    gIy2 = gaussianFilter2D(Iy2,sigmaI)
    gIxIy = gaussianFilter2D(IxIy,sigmaI)
    DgIx2 = (sigmaD**2)*gIx2
    DgIy2 = (sigmaD**2)*gIy2
    DgIxIy = (sigmaD**2)*gIxIy
    '''showImage(Ix2)
    showImage(Iy2)
    showImage(IxIy)
    showImage(gIx2)
    showImage(gIy2)
    showImage(gIxIy)
    showImage(DgIx2)
    showImage(DgIy2)
    showImage(DgIxIy)'''
#-------------------------------------------------#
#cornerHarris(loadImage('lena_gray.bmp'),0.625,1,100)

def funcion1(inputImage):
    ha = histAdapt(inputImage,92,164)
    output = histEnhance(ha,128,32)
    fig = plt.figure()
    a=fig.add_subplot(121)
    a.imshow(inputImage,vmin = 0, vmax = 255, cmap= plt.get_cmap('gray'))
    a=fig.add_subplot(122)
    a.imshow(output,vmin = 0, vmax = 255,cmap= plt.get_cmap('gray'))
    plt.show()
    return output

#funcion1(loadImage('histogram2.png'))

testEdgeCanny('canny2.png',1,0,0)

image = np.zeros((257,257))
image[(int((image.shape[0]-1)/2)),(int((image.shape[1]-1)/2))] = 1
def logFilter(inputImage,sigma):
    g = gaussianFilter2D(inputImage,sigma)
    gx,gy = derivatives(g,'CentralDiff')
    output = (gx**2)+(gy**2)
    fig = plt.figure()
    a=fig.add_subplot(121)
    a.imshow(inputImage, cmap= plt.get_cmap('gray'))
    a=fig.add_subplot(122)
    a.imshow(output,cmap= plt.get_cmap('gray'))
    plt.show()
    return output

#logFilter(image,20)
