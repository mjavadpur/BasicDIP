import cv2

def superResCUBIC(image, scale_factor:int = 3):
    
    # Upsample the image using bicubic interpolation
    img_hr = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    return img_hr


def superResAREA(image, scale_factor:int = 3):
    
    # Upsample the image using area interpolation
    img_hr = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    
    return img_hr


def superResBITS(image, scale_factor:int = 3):
    
    # Upsample the image using area interpolation
    img_hr = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_BITS)
    
    return img_hr


def superResBITS2(image, scale_factor:int = 3):
    
    # Upsample the image using area interpolation
    img_hr = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_BITS2)
    
    return img_hr


def superResLANCZOS4(image, scale_factor:int = 3):
    
    # Upsample the image using area interpolation
    img_hr = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LANCZOS4)
    
    return img_hr


def superResLINEAR(image, scale_factor:int = 3):
    
    # Upsample the image using area interpolation
    img_hr = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    
    return img_hr


def superResLINEAR_EXACT(image, scale_factor:int = 3):
    
    # Upsample the image using area interpolation
    img_hr = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR_EXACT)
    
    return img_hr



def superResNEAREST(image, scale_factor:int = 3):
    
    # Upsample the image using area interpolation
    img_hr = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
    
    return img_hr

def superResNEAREST_EXACT(image, scale_factor:int = 3):
    
    # Upsample the image using area interpolation
    img_hr = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST_EXACT)
    
    return img_hr


