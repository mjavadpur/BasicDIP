import numpy as np
import cv2

def fftshiftRGB(image):
    '''
    This method:
    1. Convert BGR image to RGB 
    2. Then compute the nD fft of the image using NumPy's fftn function
    3. Then Shift the zero-frequency component to the center 
    4. at last return shifted component in RGB format
    
    Usage
    --------
    For using the output, you must write this codes and then show the image:
    magnitude_spectrum = visibleFactory * np.log(np.abs(fshiftRGB)).astype(np.uint8)
    magnitude_spectrum =  cv2.cvtColor(magnitude_spectrum.astype(np.uint8), cv2.COLOR_RGB2BGR)  
    
    When we compute the magnitude spectrum using NumPy's abs and log functions, 
    and scale it by a factor of visibleFactory (like 20) to make it more visible. 
    '''
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    f = np.fft.fftn(img)

    fshiftRGB = np.fft.fftshift(f)
    
    return fshiftRGB

def ifftshiftBGR(fshift):
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifftn(f_ishift)
    img_back = np.abs(img_back)
    return img_back

def fftshift(image):
    imgB, imgG, imgR = cv2.split(image)
    
    fb = np.fft.fft2(imgB)
    fshiftB = np.fft.fftshift(fb)
            
    fg = np.fft.fft2(imgG)
    fshiftG = np.fft.fftshift(fg)
    
    fr = np.fft.fft2(imgR)
    fshiftR = np.fft.fftshift(fr)
    
    return fshiftB, fshiftG, fshiftR



def ifftshift(fshiftB, fshiftG, fshiftR):
    f_ishiftB = np.fft.ifftshift(fshiftB)
    img_backB = np.fft.ifftn(f_ishiftB)
    img_backB = np.abs(img_backB)
    
    f_ishiftG = np.fft.ifftshift(fshiftG)
    img_backG = np.fft.ifftn(f_ishiftG)
    img_backG = np.abs(img_backG)
    
    f_ishiftR = np.fft.ifftshift(fshiftR)
    img_backR = np.fft.ifftn(f_ishiftR)
    img_backR = np.abs(img_backR)
    
    img_back = cv2.merge([img_backB, img_backG, img_backR]).astype(int)
    return img_back

def create_mask(image, radius):
    base = np.zeros(image.shape[:2])
    cv2.circle(base, (image.shape[1]//2, image.shape[0]//2), int(radius), (1, 1, 1), -1, 8, 0)
    return base

def low_pass_filter(image, radius):
    fshiftB, fshiftG, fshiftR = fftshift(image=image)
    
    mask = create_mask(radius)
    
    fshiftB = fshiftB * mask
    
    fshiftG = fshiftG * mask
    
    fshiftR = fshiftR * mask
    ifshift = ifftshift(fshiftB, fshiftG, fshiftR)
    ifshift = cv2.normalize(ifshift, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return ifshift


def high_pass_filter(image, radius):
    
    fshiftB, fshiftG, fshiftR = fftshift(image)
    
    mask = create_mask(radius)
    
    fshiftB = fshiftB * (1 - mask)
    fshiftG = fshiftG * (1 - mask)
    fshiftR = fshiftR * (1 - mask)
    ifshift = ifftshift(fshiftB, fshiftG, fshiftR)
    ifshift = cv2.normalize(ifshift, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return ifshift
    
    

def band_pass_filter(image, min_radius, max_radius):
    
    
    
    fshiftB, fshiftG, fshiftR = fftshift(image)
    
    min_mask = create_mask(min_radius)
    max_mask = create_mask(max_radius)
    band_mask = max_mask - min_mask
    
    fshiftB = fshiftB * band_mask
    fshiftG = fshiftG * band_mask
    fshiftR = fshiftR * band_mask
    ifshift = ifftshift(fshiftB, fshiftR, fshiftR)
    ifshift = cv2.normalize(ifshift, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return ifshift
    
    

def band_reject_filter(image, min_radius, max_radius):
    
    fshiftB, fshiftG, fshiftR = fftshift(image)
    
    min_mask = create_mask(min_radius)
    max_mask = create_mask(max_radius)
    band_mask = max_mask - min_mask
    
    fshiftB = np.multiply(fshiftB , (1 - band_mask))
    fshiftG = np.multiply(fshiftG , (1 - band_mask))
    fshiftR = np.multiply(fshiftR , (1 - band_mask))
    ifshift = ifftshift(fshiftB, fshiftG, fshiftR)
    ifshift = cv2.normalize(ifshift, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return ifshift

def wiener_filtering(image, h, K):
    '''
    :param input_signal: 
    :param h: 
    :param K:
    :return:
    '''
    output_signal = [] 

    output_signal_fft = []

    input_signal_cp = np.copy(image) 

    input_signal_cp_fft = np.fft.fft2(input_signal_cp) 

    h_fft = np.fft.fft2(h) 

    h_abs_square = np.abs(h_fft)**2 

    output_signal_fft = np.conj(h_fft) / (h_abs_square + K)

    output_image = np.abs(np.fft.ifft2(output_signal_fft * input_signal_cp_fft)) 

    return output_image
