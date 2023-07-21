import numpy as np
import cv2


def low_pass_filter(image, filter_size:int = 3):
    
    if len(image.shape) <3:
        ValueError("your image must be in RGB color space!")
    img = image
    new_img= np.zeros((img.shape[0], img.shape[1], 3))
    
    filter_=  np.ones((filter_size, filter_size))
    mod= filter_size**2
    
    for i in range(filter_.shape[0]//2, new_img.shape[0]-filter_.shape[0]//2):
        for j in range(filter_.shape[1]//2, new_img.shape[1]-filter_.shape[1]//2):
            
            x1= i-filter_.shape[0]//2
            x2= x1+ filter_.shape[0]
            y1= j- filter_.shape[1]//2
            y2= y1+ filter_.shape[1]
            
            arr0= img[ x1:x2 , y1:y2, 0]
            arr1= img[ x1:x2 , y1:y2, 1 ]
            arr2= img[ x1:x2 , y1:y2, 2 ]
            
            new_img[i][j][0]= int(np.dot(np.ravel(arr0), np.ravel(filter_).T )//mod)
            new_img[i][j][1]= int(np.dot(np.ravel(arr1), np.ravel(filter_).T )//mod)
            new_img[i][j][2]= int(np.dot(np.ravel(arr2), np.ravel(filter_).T )//mod)
    return np.array(new_img, dtype= 'uint8')

def high_pass_filter(image, filter_size:int = 3):
    img = image
    new_img= np.zeros((img.shape[0], img.shape[1], 3))
    
    filter_=  -1* np.ones((filter_size, filter_size))
    mod= filter_size**2
    filter_[filter_size//2][filter_size//2]= mod- 1
    
    
    for i in range(filter_.shape[0]//2, new_img.shape[0]-filter_.shape[0]//2):
        for j in range(filter_.shape[1]//2, new_img.shape[1]-filter_.shape[1]//2):
            
            x1= i-filter_.shape[0]//2
            x2= x1+ filter_.shape[0]
            y1= j- filter_.shape[1]//2
            y2= y1+ filter_.shape[1]
            
            arr0= img[ x1:x2 , y1:y2, 0]
            arr1= img[ x1:x2 , y1:y2, 1 ]
            arr2= img[ x1:x2 , y1:y2, 2 ]
            
            new_img[i][j][0]= int(np.dot(np.ravel(arr0), np.ravel(filter_).T )//mod)
            new_img[i][j][1]= int(np.dot(np.ravel(arr1), np.ravel(filter_).T )//mod)
            new_img[i][j][2]= int(np.dot(np.ravel(arr2), np.ravel(filter_).T )//mod)
    
            if new_img[i][j][0]<0:
                new_img[i][j][0]= 0
            elif new_img[i][j][0]> 255:
                new_img[i][j][0]= 255
            
            if new_img[i][j][1]<0:
                new_img[i][j][1]= 0
            elif new_img[i][j][1]> 255:
                new_img[i][j][1]= 255
            
            if new_img[i][j][2]<0:
                new_img[i][j][2]= 0
            elif new_img[i][j][2]> 255:
                new_img[i][j][2]= 255
                
    return np.array(new_img, dtype= 'uint8')

def high_boost_filter(image, filter_size:int = 3, const: int = 5):
    img = image
    new_img= np.zeros((img.shape[0], img.shape[1], 3))
    
    filter_=  -1*np.ones((filter_size, filter_size))
    mod= filter_size**2
    filter_[filter_size//2][filter_size//2]= (mod-1)*const
    
    
    for i in range(filter_.shape[0]//2, new_img.shape[0]-filter_.shape[0]//2):
        for j in range(filter_.shape[1]//2, new_img.shape[1]-filter_.shape[1]//2):
            
            x1= i-filter_.shape[0]//2
            x2= x1+ filter_.shape[0]
            y1= j- filter_.shape[1]//2
            y2= y1+ filter_.shape[1]
            
            arr0= img[ x1:x2 , y1:y2, 0]
            arr1= img[ x1:x2 , y1:y2, 1 ]
            arr2= img[ x1:x2 , y1:y2, 2 ]
            
            new_img[i][j][0]= np.dot(np.ravel(arr0), np.ravel(filter_).T )/mod
            new_img[i][j][1]= np.dot(np.ravel(arr1), np.ravel(filter_).T )/mod
            new_img[i][j][2]= np.dot(np.ravel(arr2), np.ravel(filter_).T )/mod
    
            if new_img[i][j][0]<0:
                new_img[i][j][0]= 0
            elif new_img[i][j][0]> 255:
                new_img[i][j][0]= 255
            
            if new_img[i][j][1]<0:
                new_img[i][j][1]= 0
            elif new_img[i][j][1]> 255:
                new_img[i][j][1]= 255
            
            if new_img[i][j][2]<0:
                new_img[i][j][2]= 0
            elif new_img[i][j][2]> 255:
                new_img[i][j][2]= 255
                
    return np.array(new_img, dtype= 'uint8')

def normal_val(x, y, sig):
    val= ((x**2)+ (y**2))/2.0
    val= val/(sig**2)
    fin_val= np.exp(-val)
    return fin_val


def gaussian_filter(image, filter_size = 3, variance = 0.1) :
    n= filter_size
    sig= np.sqrt(variance)
    
    img = image
    
    filter_= [np.array([ 0.0 for i in range(-(n//2), n//2 +1)]) for j in range(-(n//2), n//2 +1)]
    
    c= 1/normal_val(n//2, n//2, sig)
    for i in range(-(n//2), n//2 +1):
        for j in range(-(n//2), n//2 +1):
            filter_[i+ n//2][j+ n//2]= round(normal_val(i, j, sig)*c, 0)
    filter_= np.array(filter_)
    
    new_img= np.zeros((img.shape[0], img.shape[1], 3))
    mod= np.sum(filter_)
    for i in range(filter_.shape[0]//2, new_img.shape[0]-filter_.shape[0]//2):
        for j in range(filter_.shape[1]//2, new_img.shape[1]-filter_.shape[1]//2):

            x1= i-filter_.shape[0]//2
            x2= x1+ filter_.shape[0]
            y1= j- filter_.shape[1]//2
            y2= y1+ filter_.shape[1]

            arr0= img[ x1:x2 , y1:y2, 0]
            arr1= img[ x1:x2 , y1:y2, 1 ]
            arr2= img[ x1:x2 , y1:y2, 2 ]

            new_img[i][j][0]= int(np.dot(np.ravel(arr0), np.ravel(filter_).T )//mod)
            new_img[i][j][1]= int(np.dot(np.ravel(arr1), np.ravel(filter_).T )//mod)
            new_img[i][j][2]= int(np.dot(np.ravel(arr2), np.ravel(filter_).T )//mod)
    
    return np.array(new_img, dtype= 'uint8')


def mean_filter(image, filter_size:int = 3, padding=0):
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
    kernel_size = (filter_size, filter_size)
    return cv2.blur(padded_image, kernel_size)

def median_filter(image, filter_size:int = 3, padding=0):
    padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
    return cv2.medianBlur(padded_image, filter_size)

def adaptive_filter(image, filter_size = 3, padding=0, maxVal = 255, \
                    adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                    thresholdType = cv2.THRESH_BINARY, \
                    borderType = cv2.BORDER_CONSTANT, \
                    C = 1):
    
    imgB, imgG, imgR = cv2.split(image)
    padded_imageB = cv2.copyMakeBorder(imgB, padding, padding, padding, padding, borderType)
    f_b = cv2.adaptiveThreshold(padded_imageB, maxVal, adaptiveMethod, thresholdType, filter_size, C)
    
    padded_imageG = cv2.copyMakeBorder(imgG, padding, padding, padding, padding, borderType)
    f_g = cv2.adaptiveThreshold(padded_imageG, maxVal, adaptiveMethod, thresholdType, filter_size, C)
    
    padded_imageR = cv2.copyMakeBorder(imgR, padding, padding, padding, padding, borderType)
    f_r = cv2.adaptiveThreshold(padded_imageR, maxVal, adaptiveMethod, thresholdType, filter_size, C)
    
    return cv2.merge([f_b, f_g, f_r])
    # padded_image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
    # return cv2.adaptiveThreshold(padded_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, filter_size, 2)

