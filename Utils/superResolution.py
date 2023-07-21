import cv2

def superResOpenCV(image, scale_factor:int = 3):
    
    # Define the scaling factor
    scale_factor = 3

    # Upsample the image using bicubic interpolation
    img_hr = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    return img_hr